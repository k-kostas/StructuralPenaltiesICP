import logging
import torch
import numpy as np
import pandas as pd
import warnings

from typing import Union, List
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from mulaconf.icp_predictor import InductiveConformalPredictor
from mulaconf.prediction_regions import PredictionRegions
from mulaconf.utils import _check_multihot_labels, _fingerprint_model, _normalize_device, _is_tensor

InputData = Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, List, float]

logger = logging.getLogger(__name__)


class ICPWrapper:
    """
    A wrapper for Inductive Conformal Prediction with Structural Penalties (Scikit-Learn compatible).

    This class manages the lifecycle of the underlying multi-label classifier and the
    conformal predictor. It handles model training, calibration, and the efficient
    integration of the PyTorch mathematical engine without unnecessary retraining.


    .. note::
        **Switching Strategies:**
        You can switch the classification strategy or update its parameters. If the wrapper detects
        a change (via fingerprinting) during calibrate, it will automatically retrain the new model
        on the cached proper-training data.


    .. note::
        **On-the-Fly Updates:**
        The wrapper itself acts strictly as a bridge. If you want to perform lazy-evaluation
        updates on the distance measure or penalty weights without passing calibration data
        again, you can do so directly via `wrapper`
        (e.g., ``wrapper.measure='norm'``, ``wrapper.weight_hamming=1.0`` and ``wrapper.weight_cardinality=0.5``).


    Parameters
    ----------
    classification_strategy : sklearn.base.BaseEstimator
        The underlying multi-label classification model (e.g., RandomForest, ClassifierChain).
        Must support ``fit`` and ``predict_proba``.
    measure : str, optional, default='mahalanobis'
        The distance metric used to score predictions.
        Supported options: ``'mahalanobis'`` (accounts for correlations) or ``'norm'`` (standard Euclidean).
    weight_hamming : float, optional, default=0.0
        Initial weight for the Hamming distance penalty.
    weight_cardinality : float, optional, default=0.0
        Initial weight for the Cardinality penalty.
    device : str or torch.device, optional, default='cpu'
        The device to use for tensor computations (``'cpu'`` or ``'cuda'``).
    """

    def __init__(self,
                 classification_strategy,
                 measure: str = 'mahalanobis',
                 weight_hamming: float = 0.0,
                 weight_cardinality: float = 0.0,
                 device: Union[str, torch.device] = 'cpu'
                 ):
        self.strategy = classification_strategy
        self.device = _normalize_device(device)

        self.icp = None

        self.measure = measure

        self.weight_hamming = float(weight_hamming)
        self.weight_cardinality = float(weight_cardinality)

        self.strategy_fingerprint = None
        self.kwargs = {}

        self.proper_train_features = None
        self.proper_train_labels = None

    @property
    def strategy(self):
        """
        Getter for the classification strategy.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy):
        """
        Setter for the classification strategy.
        Resets state to require refitting and recalibration.
        """

        self._strategy = new_strategy
        self.kwargs = {}

        self.strategy_fingerprint = None

        self.icp = None


    @property
    def measure(self) -> str:
        """
        Getter for the distance measure.
        """

        if self.icp is not None:
            return self.icp.measure
        return self._measure

    @measure.setter
    def measure(self, value: str):
        """
        Setter for the distance measure. Forwards to the InductiveConformalPredictor if fitted.
        """

        self._measure = str(value).lower().strip()
        if self.icp is not None:
            self.icp.measure = self._measure


    @property
    def weight_hamming(self) -> float:
        """
        Getter for the Hamming penalty weight.
        """

        if self.icp is not None:
            return self.icp.weight_hamming
        return self._weight_hamming

    @weight_hamming.setter
    def weight_hamming(self, value: float):
        """
        Setter for the Hamming penalty weight. Forwards to the InductiveConformalPredictor if fitted.
        """

        self._weight_hamming = float(value)
        if self.icp is not None:
            self.icp.weight_hamming = self._weight_hamming


    @property
    def weight_cardinality(self) -> float:
        """
        Getter for the Cardinality penalty weight.
        """

        if self.icp is not None:
            return self.icp.weight_cardinality
        return self._weight_cardinality

    @weight_cardinality.setter
    def weight_cardinality(self, value: float):
        """
        Setter for the Cardinality penalty weight. Forwards to the InductiveConformalPredictor if fitted.
        """

        self._weight_cardinality = float(value)
        if self.icp is not None:
            self.icp.weight_cardinality = self._weight_cardinality


    @property
    def has_pending_updates(self) -> bool:
        """
        Returns ``True`` if the distance measure or penalty weights have been updated
        and are waiting to be recalibrated in the InductiveConformalPredictor class.
        """

        if self.icp is not None:
            return (getattr(self.icp, '_update_measure', False) or
                    getattr(self.icp, '_update_weight_hamming', False) or
                    getattr(self.icp, '_update_weight_cardinality', False))
        return False


    def predict_proba_to_tensor(self, features: InputData) -> torch.Tensor:
        """
        Predicts probabilities and converts them to a unified Tensor format.

        This method handles different output formats from Scikit-Learn classifiers (e.g., standard arrays
        vs. list of arrays from ``MultiOutputClassifier``) and ensures the output is a single
        tensor of shape ``(n_samples, c_classes)``.

        Parameters
        ----------
        features : array-like
            The input features for prediction. Shape: (n_samples, w_features).

        Returns
        -------
        torch.Tensor
            A tensor containing the predicted probabilities for the positive class (1).
            Shape: (n_samples, c_classes).

        Raises
        ------
        RuntimeError
            If the underlying classification strategy has not been fitted.


        Example
        -------
        >>> # 1. Initialize and Fit Wrapper
        >>> # Load data (X_train, y_train) and model
        >>> wrapper = ICPWrapper(model)
        >>> wrapper.fit(X_train, y_train)
        >>>
        >>> # 2. Convert Probabilities to Tensor
        >>> # Internally, this handles the list conversion and any single-class edge cases.
        >>> probs = wrapper.predict_proba_to_tensor(X_train)
        """

        if torch.is_tensor(features):
            features = check_array(features.detach().cpu().numpy(), accept_sparse=True, dtype=None, ensure_2d=True)
        else:
            features = check_array(features, accept_sparse=True, dtype=None, ensure_2d=True)

        try:
            check_is_fitted(self.strategy)
        except NotFittedError:
            raise RuntimeError("Classifier has not been fitted yet. Please call fit() first.")

        probs = self.strategy.predict_proba(features)
        if isinstance(probs, list):
            extracted_probs = []
            for i, p in enumerate(probs):
                if p.shape[1] == 2:
                    extracted_probs.append(p[:, 1])
                elif p.shape[1] == 1:
                    warnings.warn(
                        "One of the labels has only 1 class. Getting the predicted class from the classifier.",
                        RuntimeWarning)
                    present_class = self.strategy.classes_[i][0]
                    if present_class == 0:
                        extracted_probs.append(np.zeros_like(p[:, 0]))
                    else:
                        extracted_probs.append(p[:, 0])
                else:
                    extracted_probs.append(p[:, 1])
            probs = np.array(extracted_probs).T

        return torch.tensor(probs, device=self.device, dtype=torch.float32)


    def fit(self, train_features: InputData, train_labels: InputData, **kwargs):
        """
        Fits the underlying multi-label classification model.

        This method trains the ``classification_strategy`` on the provided ``features`` and ``labels``,
        and caches the training data to enable auto-retraining if hyper-parameters change.

        Parameters
        ----------
        train_features : array-like
            The training features. Shape: (n_samples, w_features).
        train_labels : array-like
            The training labels (binary multi-hot). Shape: (n_samples, c_classes).
        **kwargs : dict
            Optional arguments to pass to the classifier's parameters or ``fit`` method.

        Returns
        -------
        self : object
            The fitted wrapper instance.


        Example
        --------
        >>> import numpy as np
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.multioutput import MultiOutputClassifier
        >>>
        >>> # 1. Generate Dummy Training Data
        >>> # 100 samples, 5 features, 3 target labels
        >>> X_train = np.random.rand(100, 5)
        >>> y_train = np.random.randint(0, 2, (100, 3))
        >>>
        >>> # 2. Initialize Wrapper
        >>> base_model = MultiOutputClassifier(RandomForestClassifier())
        >>> wrapper = ICPWrapper(base_model)
        >>>
        >>> # 3. Fit the Model (Standard)
        >>> wrapper.fit(X_train, y_train)

        .. note::
            **Change hyperparameters** : You can update the classifier's hyperparameters during the fit call.
            Note the ``estimator__`` prefix for wrapped sklearn models.

            >>> args = {'estimator__n_neighbors': 5}
            >>> wrapper.fit(X_train, y_train, **args)
        """

        logger.info("Starting fit procedure...")
        if train_features is not None and train_labels is not None:
            train_labels = _check_multihot_labels(train_labels)
            if torch.is_tensor(train_labels):
                train_labels = check_array(train_labels.detach().cpu().numpy(), ensure_2d=False, allow_nd=True)
            else:
                train_labels = check_array(train_labels, ensure_2d=False, allow_nd=True)

            if torch.is_tensor(train_features):
                train_features = check_array(train_features.detach().cpu().numpy(), accept_sparse=True, dtype=None,
                                             ensure_2d=True)
            else:
                train_features = check_array(train_features, accept_sparse=True, dtype=None, ensure_2d=True)
        else:
            raise ValueError("Both train_features and train_labels must be provided for fitting.")

        if kwargs:
            self.strategy.set_params(**kwargs)
            self.kwargs = kwargs
        else:
            self.kwargs = {}

        logger.info("Fitting classifier...")
        self.strategy.fit(train_features, train_labels)

        self.strategy_fingerprint = _fingerprint_model(self.strategy, self.kwargs)
        self.proper_train_features = train_features
        self.proper_train_labels = train_labels

        self.icp = InductiveConformalPredictor(
            predicted_probabilities=self.predict_proba_to_tensor(self.proper_train_features).to(self.device),
            true_labels=_is_tensor(self.proper_train_labels).to(self.device),
            measure=self.measure,
            weight_hamming=self.weight_hamming,
            weight_cardinality=self.weight_cardinality,
            device=self.device
        )

        logger.info("Classifier trained with features shape: %s", train_features.shape)
        logger.info("Fit complete.")

        return self


    def calibrate(self, calib_features: InputData = None, calib_labels: InputData = None):
        """
        Calibrates the conformal predictor using a dedicated calibration set.

        This step calculates the nonconformity scores and determines the thresholds required to guarantee
        coverage.

        .. note::
            If called without arguments after an initial calibration, it will manually
            apply any pending updates (measure or penalty weights) and recalibrate
            using the cached data.

        Parameters
        ----------
        calib_features : array-like
            Features of the calibration set. Shape: (q_samples, w_features).
        calib_labels : array-like
            Labels of the calibration set. Shape: (q_samples, c_classes).

        Returns
        -------
        self : object
            The calibrated wrapper instance.

        Raises
        ------
        RuntimeError
            If calibration features and labels are not provided.
        RuntimeError
            If ``fit()`` has not been called before running calibration.
        RuntimeError
            If retraining the underlying classifier fails.


        Example
        --------
        >>> # 1. Initialize & Fit (See `fit()` function documentation for details)
        >>>
        >>> # 2. Generate Dummy Calibration Data
        >>> # 100 samples, 5 features, 3 target labels
        >>> X_calib = np.random.rand(100, 5)
        >>> y_calib = np.random.randint(0, 2, (100, 3))
        >>>
        >>> # Calibrate
        >>> wrapper.calibrate(X_calib, y_calib)


        .. note::
            **Strategy Switching**: Change the underlying model, retrain and recalibrate automatically.

            >>> from sklearn.neighbors import KNeighborsClassifier
            >>> from sklearn.multioutput import ClassifierChain
            >>>
            >>> wrapper.strategy = ClassifierChain(KNeighborsClassifier(n_neighbors=5))
            >>>
            >>> # Calling `calibrate()` again detects the change and retrains automatically
            >>> wrapper.calibrate(X_calib, y_calib)

        .. note::
            **On-the-fly Updates**: You can easily update the distance measure and penalty weights after
            the calibration process without passing your data again. Calling `calibrate()` without arguments will
            automatically apply all pending updates simultaneously using the cached calibration data.

            >>> # Optional: The `calibrate()` method recalculates the covariance matrix and scores after a measure update.
            >>> wrapper.measure = 'norm'
            >>> wrapper.calibrate()


            >>> # Optional: The `calibrate()` method recalculates calibration scores after penalty weight update.
            >>> wrapper.weight_hamming = 1.0
            >>> wrapper.weight_cardinality = 0.5
            >>> wrapper.calibrate()


            >>> # Optional: The `calibrate()` method applies both distance measure and penalty weight updates at once.
            >>> wrapper.measure = 'norm'
            >>> wrapper.weight_hamming = 1.0
            >>> wrapper.weight_cardinality = 0.5
            >>> wrapper.calibrate()
        """

        logger.info("Starting calibration...")

        if self.proper_train_features is None or self.proper_train_labels is None:
            raise RuntimeError("Run the fit() procedure first. Proper training data is missing.")

        if calib_features is None and calib_labels is None:
            if self.icp is None or getattr(self.icp, 'calib_probabilities', None) is None or getattr(self.icp,
                                                                                                     'calib_labels',
                                                                                                     None) is None:
                raise RuntimeError("No cached calibration data. Please provide calib_features and calib_labels first.")

            if not self.has_pending_updates:
                logger.info("No updates detected. Predictor is already calibrated.")
                return self

            logger.info("Pending updates detected. Recalibrating on cached data...")

            self.icp.calibrate()
            logger.info("Calibration complete.")
            return self

        if calib_features is not None and calib_labels is not None:
            calib_labels = _check_multihot_labels(calib_labels)

            if torch.is_tensor(calib_features):
                calib_features = check_array(calib_features.detach().cpu().numpy(), accept_sparse=True, dtype=None,
                                             ensure_2d=True)
            else:
                calib_features = check_array(calib_features, accept_sparse=True, dtype=None, ensure_2d=True)
        else:
            raise RuntimeError("Calibration features and labels must be provided to calibrate().")

        try:
            check_is_fitted(self.strategy)
            is_fitted = True
        except NotFittedError:
            is_fitted = False

        if not is_fitted or self.strategy_fingerprint is None or self.strategy_fingerprint != _fingerprint_model(
                self.strategy, self.kwargs):
            logger.info("Classifier model change detected. Retraining the classifier...")
            try:
                self.strategy.fit(self.proper_train_features, self.proper_train_labels)
                check_is_fitted(self.strategy)
                self.strategy_fingerprint = _fingerprint_model(self.strategy, self.kwargs)

                self.icp = InductiveConformalPredictor(
                    predicted_probabilities=self.predict_proba_to_tensor(self.proper_train_features).to(self.device),
                    true_labels=_is_tensor(self.proper_train_labels).to(self.device),
                    measure=self.measure,
                    weight_hamming=self.weight_hamming,
                    weight_cardinality=self.weight_cardinality,
                    device=self.device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to retrain classifier with new parameters: {e}")

        if self.icp is None:
            raise RuntimeError("Run the fit() procedure first.")

        self.icp.calibrate(self.predict_proba_to_tensor(calib_features).to(self.device),
                           _is_tensor(calib_labels).to(self.device))
        logger.info("Calibration complete.")

        return self


    def predict(self, test_features: InputData) -> PredictionRegions:
        """
        Generates conformal prediction regions for the input features.

        This method calculates p-values for all test samples based on the calibrated scores.

        Parameters
        ----------
        test_features : array-like
            The test features. Shape: (t_samples, w_features).


        Returns
        -------
        PredictionRegions
            A callable object that wraps the p-values and label-set combinations.
            The object can be called with a specific ``significance_level`` to extract prediction regions.
            It also exposes the ``non_empty_prediction_regions`` property, which controls whether
            empty prediction regions are corrected by adding the maximum-p-value label-set.


        Raises
        ------
        RuntimeError
            If ``calibrate()`` has not been called, and the ICP engine does not exist.
        RuntimeError
            If the classifier has not been fitted yet (``fit()`` must be called first).
        RuntimeError
            If the classifier model changed. Run the fit and calibration procedure.


        Example
        --------
        >>> # ... Assume wrapper is already fitted and calibrated (see fit() for details) ...
        >>> X_test = np.random.rand(10, 5)
        >>>
        >>> # 1. Get a PredictionRegions object. By default, non-empty prediction regions are enabled.
        >>> prediction_obj = wrapper.predict(X_test)
        >>>
        >>> # 2. Extract Prediction Sets (e.g., at 10% significance / 90% confidence)
        >>> # Returns a list of Tensors, where each Tensor contains the indices of predicted labels.
        >>> prediction_regions = prediction_obj(significance_level=0.1)


        .. note::
            **Equivalent Syntax**: Because the ``predict`` method returns a callable
            ``PredictionRegions`` object, you can chain the operations to evaluate
            the test features and extract prediction sets in a single line of code:

            >>> prediction_regions = wrapper.predict(X_test)(significance_level=0.1)


        .. note::
            **On-the-fly Update**: You can update the distance measure and penalty weights on the fly and
            predict again immediately.

            >>> wrapper.measure = 'norm'
            >>> wrapper.weight_hamming = 2.0
            >>> wrapper.weight_cardinality = 1.5
            >>> updated_obj = wrapper.predict(X_test)
            >>> updated_regions = updated_obj(significance_level=0.1)


        .. note::
            The returned ``PredictionRegions`` object controls whether prediction regions can be empty
            through its ``non_empty_prediction_regions`` property. If ``True`` (the default), the object
            includes the label-set with the highest p-value whenever no label-set satisfies the selected
            significance threshold. This correction can be disabled before extracting prediction regions.

            >>> prediction_obj = wrapper.predict(X_test)
            >>> prediction_obj.non_empty_prediction_regions = False
            >>> prediction_regions = prediction_obj(significance_level=0.1)
        """

        logger.info("Starting prediction...")
        if self.icp is None:
            raise RuntimeError("Run the calibrate() procedure first.")

        try:
            check_is_fitted(self.strategy)
        except NotFittedError:
            raise RuntimeError("Classifier must be fitted.")

        if self.strategy_fingerprint != _fingerprint_model(self.strategy, self.kwargs):
            raise RuntimeError("Classifier model changed. Run the fit and calibration procedure.")

        if torch.is_tensor(test_features):
            test_features = check_array(test_features.detach().cpu().numpy(), accept_sparse=True, dtype=None,
                                        ensure_2d=True)
        else:
            test_features = check_array(test_features, accept_sparse=True, dtype=None, ensure_2d=True)

        test_probabilities = self.predict_proba_to_tensor(test_features).to(self.device)
        logger.debug("Returning PredictionRegions object.")

        return self.icp.predict(test_probabilities)