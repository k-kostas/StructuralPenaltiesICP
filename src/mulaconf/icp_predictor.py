import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Union

from .prediction_regions import PredictionRegions
from .utils import _check_multihot_labels, _is_tensor, _normalize_device
from . import constants

InputData = Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]

from sklearn.covariance import ledoit_wolf

class InductiveConformalPredictor:
    """
    Inductive Conformal Predictor with Structural Penalties.

    This class implements Inductive Conformal Prediction (ICP) for multi-label classification, extended with
    structural penalties (Hamming and Cardinality). It uses a generalized distance metric (e.g., Mahalanobis
    or Euclidean Norm) in the error vector space to score predictions. Additionally, it allows for on-the-fly
    updating of the distance measure and penalty weights without retraining the underlying model or requiring
    the calibration data to be passed again.

    .. note::
       The predictor calculates and caches the structural penalty vectors for all possible label
       combinations during initialization.

    .. note::
       **On-the-Fly Updates**:
       You can update the distance measure (``'norm'`` or ``'mahalanobis'``), ``weight_hamming``,
       and ``weight_cardinality`` at any time after the calibration process.

       The predictor utilizes lazy evaluation for automatic recalibration. This means
       you do not need to manually pass your calibration data again or explicitly call
       ``calibrate()``. Simply assign new values to the properties (e.g., ``icp.measure = 'norm'``,
       ``icp.weight_hamming = 1.0``, ``icp.weight_cardinality = 0.5``) and immediately call ``predict()``. It will
       automatically reform the underlying covariance matrix and recalibrate the scores
       on the fly before generating predictions.


    .. note::
       **Memory Management**:
       This class uses optimized batching, tensor expansion (compressed loading) to prevent GPU/CPU memory
       overflow when processing exponential
       powerset combinations. Even with these optimizations, Powerset Scoring prediction scales
       at O(2^C), where C is the number of labels. For standard systems with 16GB of RAM/VRAM,
       we recommend limiting tasks to a maximum of ~20 labels.

       The default batching limits are approximated based on PyTorch's
       ``float32`` data type (which consumes 4 bytes per element) and the overhead
       required to hold multiple massive intermediate tensors simultaneously in memory
       during calculation.

       Users can manually tune the hardware limits by modifying the module-level configuration
       variables (located in ``constants.py``) to optimize for their specific CPU/GPU memory constraints:

       ``_CPU_MAX_COMBINATIONS``: Caps the maximum number of combinations processed at once during
       heavy matrix math on the CPU to protect system RAM (default: 2,000,000).

       ``_GPU_MAX_COMBINATIONS``: Caps the maximum number of combinations processed at once during
       heavy matrix math on the GPU to protect VRAM (default: 5,000,000).

       ``_REGION_BATCH_SIZE``: Caps memory usage (System RAM) when extracting the final prediction sets.
       Because this phase relies on lightweight boolean filtering rather than heavy matrix operations,
       this threshold is safely set much higher (default: 100,000,000).

       ``_EMPTY_CUDA_CACHE``: Controls whether the engine aggressively clears the CUDA memory cache after
       heavy tensor operations. Keep this set to ``True`` (default) to prevent VRAM fragmentation and
       Out-Of-Memory crashes. Set to ``False`` only for strict performance benchmarking to bypass the
       ~300ms OS-level synchronization delay.

       Increasing these values speeds up the time required to generate predictions, but risks memory overflow
       and system instability. Decreasing these values results in slower predictions but guarantees safety
       from system crashes. In the documentation of ``constants.py``, you can find a hardware cheat sheet
       for memory requirements.


    Parameters
    ----------
    measure : str, optional, default='mahalanobis'
        The distance metric used to score predictions.
        Supported options: 'mahalanobis' (accounts for correlations) or 'norm' (standard Euclidean).
    predicted_probabilities : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The predicted probabilities for the proper training set.
        Shape: (n_samples, c_classes).
    true_labels : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The ground truth binary labels for the proper training set.
        Shape: (n_samples, c_classes).
    weight_hamming : float, optional, default=0.0
        The weight for the Hamming distance penalty. Higher values penalize predictions
        that are structurally different from observed training labels.
    weight_cardinality : float, optional, default=0.0
        The weight for the Cardinality penalty. Higher values penalize prediction set sizes
        that are infrequent in the training data.
    device : str or torch.device, optional, default='cpu'
        The device to use for computations ('cpu' or 'cuda').


    Example
    -------
    >>> import torch
    >>>
    >>> # 1. Generate dummy training data (500 samples, 5 classes)
    >>> train_probs = torch.rand(500, 5)
    >>> train_labels = torch.randint(0, 2, (500, 5)).float()
    >>>
    >>> # 2. Initialize the predictor
    >>> icp = InductiveConformalPredictor(
    ...     predicted_probabilities=train_probs,
    ...     true_labels=train_labels,
    ...     measure='mahalanobis',
    ...     weight_hamming=2.0,
    ...     weight_cardinality=1.5,
    ... )
    """

    def __init__(self,
                 predicted_probabilities: InputData,
                 true_labels: InputData,
                 measure: str = 'mahalanobis',
                 weight_hamming: float = 0.0,
                 weight_cardinality: float = 0.0,
                 device: Union[str, torch.device] = 'cpu',
                 ):

        print(f'\nInitializing Inductive Conformal Predictor')

        self.device = _normalize_device(device)

        true_labels = _check_multihot_labels(true_labels)
        true_labels = _is_tensor(true_labels).to(self.device)

        predicted_probabilities = _is_tensor(predicted_probabilities).to(self.device)
        if predicted_probabilities.shape[1] != true_labels.shape[1]:
            raise RuntimeError("Proper train labels and probabilities must have the same number of columns.")

        self.n_classes = true_labels.shape[1]

        self._measure = measure.lower().strip()
        if self._measure not in ['mahalanobis', 'norm']:
            raise ValueError(f"Invalid measure '{measure}'. Supported options are 'mahalanobis' or 'norm'.")

        self.matrix_power_parameter = -1.0 if self._measure == 'mahalanobis' else 0.0
        self._weight_hamming = float(weight_hamming)
        self._weight_cardinality = float(weight_cardinality)

        self.combinations = torch.cartesian_prod(
            *[torch.tensor([False, True], device=self.device)] * self.n_classes
        )

        self.proper_train_labels = true_labels
        self.proper_train_probabilities = predicted_probabilities

        self.calib_probabilities = None
        self.calib_labels = None

        self._hamming_penalties = None
        self._cardinality_penalties = None
        self._distance_matrix = None

        self._max_distance_score = None
        self._calib_normalized_scores = None
        self._calib_indices = None
        self.sorted_calibration_scores = None

        self._update_weight_hamming = False
        self._update_weight_cardinality = False
        self._update_measure = False

        self.hamming_penalties_preprocessing(self.proper_train_labels)
        self.cardinality_penalties_preprocessing(self.proper_train_labels)
        self.covariance_matrix_preprocessing(self.proper_train_probabilities, self.proper_train_labels)


    @property
    def measure(self) -> str:
        """
        Getter for the current distance measure.
        """

        return self._measure

    @measure.setter
    def measure(self, value: str):
        """
        Set the distance measure.
        Triggers a flag to rebuild the covariance matrix and recalculate calibration scores.
        """

        cleaned_value = str(value).lower().strip()
        if cleaned_value not in ['mahalanobis', 'norm']:
            raise ValueError(f"Invalid measure '{value}'. Supported options are 'mahalanobis' or 'norm'.")

        if self._measure != cleaned_value:
            self._measure = cleaned_value
            self.matrix_power_parameter = -1.0 if self._measure == 'mahalanobis' else 0.0

            self._update_measure = True
            print(f"Measure changed to '{cleaned_value}'. Flagged for recalibration.")


    @property
    def weight_hamming(self) -> float:
        """
        Getter for the current Hamming penalty weight.
        """

        return self._weight_hamming

    @weight_hamming.setter
    def weight_hamming(self, value: float):
        """
        Set the Hamming penalty weight.

        Setting this property triggers a flag to recalculate calibration scores
        during the next prediction call without re-running the
        full calibration procedure.


        .. note::
            If switching from 0.0 to a positive value, Hamming penalties are recalculated.


        Returns
        -------
        float
            The current Hamming penalty weight.
        """

        if value < 0:
            raise ValueError("Hamming penalty weight cannot be negative.")

        if self._weight_hamming != value:
            print(f'\n---Updating weight for Hamming penalties---')
            if self._weight_hamming == 0 and value > 0:
                self._weight_hamming = value
                self._update_weight_hamming = True
                self.hamming_penalties_preprocessing(self.proper_train_labels)
                print(f"Hamming penalty weight updated to {value}.")
                print(f'Hamming penalties recalculated.')
            else:
                self._weight_hamming = value
                self._update_weight_hamming = True
                print(f"Hamming penalty weight updated to {value}.")


    @property
    def weight_cardinality(self) -> float:
        """
        Getter for the current Cardinality penalty weight.
        """

        return self._weight_cardinality

    @weight_cardinality.setter
    def weight_cardinality(self, value: float):
        """
        Set the Cardinality penalty weight.

        Setting this property triggers a flag to recalculate calibration scores
        during the next prediction call without rerunning the
        full calibration procedure.

        .. note::
            If switching from 0.0 to a positive value, Cardinality penalties are recalculated.


        Returns
        -------
        float
            The current Cardinality penalty weight.
        """

        if value < 0:
            raise ValueError("Cardinality penalty weight cannot be negative.")

        if self._weight_cardinality != value:
            print(f'\n---Updating weight for Cardinality penalties---')
            if self._weight_cardinality == 0 and value > 0:
                self._weight_cardinality = value
                self._update_weight_cardinality = True
                self.cardinality_penalties_preprocessing(self.proper_train_labels)
                print(f"Cardinality penalty weight updated to {value}.")
                print(f'Cardinality penalties recalculated.')
            else:
                self._weight_cardinality = value
                self._update_weight_cardinality = True
                print(f"Cardinality penalty weight updated to {value}.")


    @torch.no_grad()
    def hamming_penalties_preprocessing(self, labels: torch.Tensor):
        """
        Calculates Hamming penalties for all possible label combinations.

        The penalty is defined as the minimum Hamming distance from a combination
        to any observed label vector in the provided labels.


        Parameters
        ----------
        labels : torch.Tensor
            The set of ground truth labels of the proper training set.
            Shape: (n_samples, c_classes).


        Example
        --------
        >>> # 1. Generate dummy data (100 samples, 5 classes)
        >>> labels = torch.randint(0, 2, (100, 5)).float()
        >>> icp.hamming_penalties_preprocessing(labels)
        """

        if self._weight_hamming == 0:
            self._hamming_penalties = torch.zeros(self.combinations.shape[0], device=self.device)
            return

        labels = labels.float().to(self.device)
        labels_sum = labels.sum(dim=1)

        n_samples = labels.shape[0]

        max_combs = constants._GPU_MAX_COMBINATIONS if str(self.device).startswith('cuda') else constants._CPU_MAX_COMBINATIONS
        batch_size = max(1, max_combs // n_samples)

        min_distances_list = []
        iterator = range(0, self.combinations.shape[0], batch_size)
        if self.combinations.shape[0] > batch_size:
            iterator = tqdm(iterator, desc="Calculating Hamming Penalties")

        for i in iterator:
            comb_batch = self.combinations[i: i + batch_size].float().to(self.device)
            comb_sum = comb_batch.sum(dim=1, keepdim=True)

            dot_product = comb_batch @ labels.T
            dot_product.mul_(-2.0)
            dot_product.add_(comb_sum)
            dot_product.add_(labels_sum)

            batch_loss = dot_product / self.n_classes
            batch_min_dists = torch.min(batch_loss, dim=1).values
            min_distances_list.append(batch_min_dists)

            del comb_batch, comb_sum, dot_product, batch_loss, batch_min_dists

        if torch.cuda.is_available() and constants._EMPTY_CUDA_CACHE:
            torch.cuda.empty_cache()

        self._hamming_penalties = torch.cat(min_distances_list)
        print("Hamming penalties calculated with shape:", self._hamming_penalties.shape)


    @torch.no_grad()
    def cardinality_penalties_preprocessing(self, labels: torch.Tensor):
        """
        Calculates Cardinality penalties based on label set size frequencies.

        Combinations with a cardinality (number of active labels) that appears frequently
        in the training data receive lower penalties.


        Parameters
        ----------
        labels : torch.Tensor
            The set of ground truth labels used to calculate size frequencies.
            Shape: (n_samples, c_classes).


        Example
        --------
        >>> # Generate dummy data (100 samples, 5 classes)
        >>> labels = torch.randint(0, 2, (100, 5)).float()
        >>> icp.cardinality_penalties_preprocessing(labels)
        """

        if self._weight_cardinality == 0:
            self._cardinality_penalties = torch.zeros(self.combinations.shape[0], device=self.device)
            return

        labels = labels.to(self.device)
        card_counts = torch.bincount(torch.sum(labels, dim=1).long(), minlength=self.n_classes + 1)
        total_counts = card_counts.sum()

        if total_counts > 0:
            penalty_lookup = 1.0 - (card_counts.float() / total_counts.float())
        else:
            penalty_lookup = torch.ones(self.n_classes + 1, dtype=torch.float, device=self.device)

        batch_size = constants._GPU_MAX_COMBINATIONS if str(self.device).startswith('cuda') else constants._CPU_MAX_COMBINATIONS
        penalties_list = []

        iterator = range(0, self.combinations.shape[0], batch_size)
        if self.combinations.shape[0] > batch_size:
            iterator = tqdm(iterator, desc="Calculating Cardinality Penalties")

        for i in iterator:
            comb_chunk = self.combinations[i: i + batch_size]
            chunk_cards = torch.sum(comb_chunk, dim=1).long()
            chunk_penalties = penalty_lookup[chunk_cards]
            penalties_list.append(chunk_penalties)

            del comb_chunk, chunk_cards, chunk_penalties

        if torch.cuda.is_available() and constants._EMPTY_CUDA_CACHE:
            torch.cuda.empty_cache()

        self._cardinality_penalties = torch.cat(penalties_list)
        print("Cardinality penalties calculated with shape:", self._cardinality_penalties.shape)


    @torch.no_grad()
    def covariance_matrix_preprocessing(self, probabilities: torch.Tensor, labels: torch.Tensor):
        """
        Computes the generalized covariance matrix for the error vectors
        (|Predicted Probabilities - Labels|) on the Proper Training Set.

        .. note::
            Covariance matrices are calculated using Ledoit-Wolf shrinkage
            to ensure positive-definiteness.

        If ``measure='mahalanobis'``, this computes the Inverse Covariance Matrix.
        If ``measure='norm'``, this effectively computes an Identity Matrix.


        Parameters
        ----------
        probabilities : torch.Tensor
            Predicted probabilities for the proper training set.
            Shape: (n_samples, c_classes).
        labels : torch.Tensor
            True labels for the proper training set.
             Shape: (n_samples, c_classes).


        Raises
        ------
        RuntimeError
            If the number of classes is less than 2. Single-class datasets are not
            supported for multi-label conformal prediction.


        Example
        --------
        >>> # Generate dummy data (100 samples, 5 classes)
        >>> probabilities = torch.rand(100, 5)
        >>> labels = torch.randint(0, 2, (100, 5)).float()
        >>> icp.covariance_matrix_preprocessing(probabilities, labels)
        """

        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)

        if probabilities.shape[1] < 2:
            raise RuntimeError(
                f"InductiveConformalPredictor requires at least 2 classes, but got {probabilities.shape[1]}. "
                "Single-class datasets are not supported for multi-label conformal prediction."
            )

        errors = torch.abs(probabilities - labels)
        errors_np = errors.cpu().numpy()
        shrunk_cov_np, optimal_alpha = ledoit_wolf(errors_np)

        covariance_matrix = torch.tensor(shrunk_cov_np, dtype=torch.float32, device=self.device)
        eigvalues, eigvectors = torch.linalg.eig(covariance_matrix)

        diagonal_covariance_matrix_power = torch.diag(eigvalues.real.pow(self.matrix_power_parameter))

        self._distance_matrix = (
                eigvectors.real @ diagonal_covariance_matrix_power @ eigvectors.real.T
        ).to(device=self.device)

        distance_matrix_abs = torch.abs(self._distance_matrix).to(device=self.device)

        ones = torch.ones(self.n_classes, device=self.device)
        self._max_distance_score = torch.sqrt(ones @ distance_matrix_abs @ ones).to(device=self.device)
        print(f"Distance matrix calculated (Measure: {self._measure}) with shape:", self._distance_matrix.shape)


    def _update_calibration_scores(self):
        """
        Updates and sorts calibration scores based on current penalty weights.
        Internal method called automatically by ``predict()`` or ``calibrate()`` if weights change.


        Raises
        ------
        RuntimeError
            If calibration scores are not initialized. Call ``calibrate()`` with calibration features probabilities and labels first.
        """

        if self._calib_normalized_scores is not None and self._calib_indices is not None:
            calibration_scores = self._calib_normalized_scores + \
                                 (self.weight_hamming * self._hamming_penalties[self._calib_indices]) + \
                                 (self.weight_cardinality * self._cardinality_penalties[self._calib_indices])

            self.sorted_calibration_scores, _ = torch.sort(calibration_scores, descending=True)
            self._update_weight_hamming = False
            self._update_weight_cardinality = False
            print("Calibration scores calculated with shape:", self.sorted_calibration_scores.shape)
        else:
            raise RuntimeError("Calibration scores are not initialized. Call calibrate() first.")


    @torch.no_grad()
    def calibrate(self, probabilities: InputData = None, labels: InputData = None):
        """
        Calibrates the predictor using a dedicated calibration set.

        This method computes nonconformity scores for the calibration data and
        sorts them to determine thresholds for future predictions.


        .. note::
            If called without arguments, it recalculates the calibration scores
            using the current distance measure and set penalty weights on the existing calibration data.


        Parameters
        ----------
        probabilities : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
            Predicted probabilities for the calibration set.
            Shape: (q_samples, c_classes).
        labels : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
            Ground truth labels for the calibration set.
            Shape: (q_samples, c_classes).


        Returns
        -------
        self : object
            The initialized and calibrated predictor object.


        Raises
        ------
        RuntimeError
            If one of ``probabilities`` or ``labels`` is provided but not the other.
        RuntimeError
            If the provided calibration set is empty.
        RuntimeError
            If ``labels`` shape does not match the number of classes.
        RuntimeError
            If ``probabilities`` shape does not match the number of classes.


        Example
        --------
        >>> # 1. Generate dummy calibration data (100 samples, 5 classes)
        >>> calib_probs = torch.rand(100, 5)
        >>> calib_labels = torch.randint(0, 2, (100, 5)).float()
        >>>
        >>> # 2. Calibrate
        >>> icp.calibrate(calib_probs, calib_labels)


        .. note::
            Update the distance measure and penalty weights before calling ``calibrate()``.

            >>> # Optional: The `calibrate()` method recalculates the distance matrix and scores after a measure update.
            >>> icp.measure = 'norm'
            >>> icp.calibrate()


            >>> # Optional: The `calibrate()` method recalculates calibration scores after penalty weight update.
            >>> icp.weight_hamming = 1.0
            >>> icp.weight_cardinality = 0.5
            >>> icp.calibrate()


            >>> # Optional: The `calibrate()` method applies both distance measure and penalty weight updates at once.
            >>> icp.measure = 'norm'
            >>> icp.weight_hamming = 1.0
            >>> icp.weight_cardinality = 0.5
            >>> icp.calibrate()
        """

        recalculate_distance_scores = False

        if getattr(self, '_update_measure', False):
            if self.proper_train_probabilities is None or self.proper_train_labels is None:
                raise RuntimeError("Cannot recalculate distance matrix: Proper training data is missing.")

            print("Applying measure update and recalculating covariance matrix...")
            self.covariance_matrix_preprocessing(self.proper_train_probabilities, self.proper_train_labels)
            self._update_measure = False
            recalculate_distance_scores = True


        if probabilities is not None and labels is not None:
            if torch.is_tensor(probabilities):
                self.calib_probabilities = probabilities.detach().clone().to(self.device)
            else:
                self.calib_probabilities = _is_tensor(probabilities).to(self.device)

            if torch.is_tensor(labels):
                self.calib_labels = labels.detach().clone().to(self.device)
            else:
                self.calib_labels = _is_tensor(_check_multihot_labels(labels)).to(self.device)


            if self.calib_probabilities.shape[1] != self.n_classes:
                raise RuntimeError("Calibration labels and probabilities must have the same number of columns.")

            if self.calib_labels.shape[0] == 0:
                raise RuntimeError("Calibration set cannot be empty.")

            if self.calib_labels.shape[1] != self.n_classes:
                raise RuntimeError("Labels must have the same number of columns as the number of classes.")

            recalculate_distance_scores = True
        elif (probabilities is None) != (labels is None):
            raise RuntimeError("Both 'probabilities' and 'labels' must be provided for calibration.")
        elif self.calib_probabilities is None or self.calib_labels is None:
            raise RuntimeError("No calibration data is cached. Please provide probabilities and labels.")

        if recalculate_distance_scores:
            if self.calib_probabilities.ndim == 1:
                probs = self.calib_probabilities.unsqueeze(0)
            else:
                probs = self.calib_probabilities

            errors = torch.abs(probs - self.calib_labels)
            distance_scores = torch.sqrt(torch.sum((errors @ self._distance_matrix) * errors, dim=1))
            self._calib_normalized_scores = distance_scores / self._max_distance_score

            powers_desc = 2 ** torch.arange(self.calib_labels.shape[1] - 1, -1, -1, device=self.device)
            self._calib_indices = (self.calib_labels * powers_desc).sum(dim=1).long()


        self._update_calibration_scores()

        return self


    def all_combinations_scoring(self, probabilities: torch.Tensor):
        """
        Computes nonconformity scores for a test sample against all possible combinations.


        Parameters
        ----------
        probabilities : torch.Tensor
            Predicted probabilities for the input sample.
            Shape: (t_samples, c_classes).


        Returns
        -------
        torch.Tensor
            A 1D tensor containing the calculated nonconformity scores for every
            possible label combination for the given test sample.
            Shape: (2^(c_classes))
        """

        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)

        probs_expanded = probabilities.unsqueeze(1)

        n_samples = probabilities.shape[0]

        max_combs = constants._GPU_MAX_COMBINATIONS if str(self.device).startswith('cuda') else constants._CPU_MAX_COMBINATIONS
        chunk_size = max(1, max_combs // n_samples)

        all_scores_list = []

        for i in range(0, self.combinations.shape[0], chunk_size):
            combs_chunk = self.combinations[i: i + chunk_size].float().to(self.device).unsqueeze(0)
            errors = torch.abs(probs_expanded - combs_chunk)
            distance_scores = torch.sqrt(torch.sum((errors @ self._distance_matrix) * errors, dim=-1))
            normalized_scores = distance_scores / self._max_distance_score

            hamming_chunk = self._hamming_penalties[i: i + chunk_size]
            cardinality_chunk = self._cardinality_penalties[i: i + chunk_size]

            chunk_scores = normalized_scores + \
                           (self.weight_hamming * hamming_chunk) + \
                           (self.weight_cardinality * cardinality_chunk)

            all_scores_list.append(chunk_scores)
            del combs_chunk, errors, distance_scores, normalized_scores, chunk_scores

        return torch.cat(all_scores_list, dim=1)


    @torch.no_grad()
    def predict(self, probabilities: InputData, non_empty_prediction_regions:bool = True) -> PredictionRegions:
        """
        Computes p-values for the test samples.

        This method calculates the p-value for every possible label combination
        based on the calibrated scores.


        Parameters
        ----------
        probabilities : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
            Predicted probabilities for the test set.
            Shape: (t_samples, c_classes).
        non_empty_prediction_regions : bool, optional
            If True (default), the combination with the highest p-value is returned to ensure non-empty predictions.


        Returns
        -------
        PredictionRegions
            A callable object that wraps the p-values and combinations.
            You must call this object with a significance level to get the actual prediction sets.


        Raises
        ------
        RuntimeError
            If a distance measure was changed, but no
            calibration data is cached to perform the auto-recalculation.
        RuntimeError
            If ``calibrate()`` has not been called before ``predict()``.
        RuntimeError
            If ``probabilities`` shape does not match the number of classes.


        Example
        --------
        >>> # Generate dummy test probabilities
        >>> test_probs = torch.rand(30, 5)
        >>>
        >>> # Get prediction regions object with non empty prediction regions
        >>> prediction_obj = icp.predict(test_probs)
        >>>
        >>> # Extract prediction sets for significance level 0.1 (90% confidence)
        >>> prediction_sets = prediction_obj(significance_level=0.1)

        .. note::
            **Equivalent Syntax**: Because the predictor itself is callable and it returns a callable
            ``PredictionRegions`` object, you can chain the operations to extract prediction sets in a single line of code:

            >>> prediction_sets = icp.predict(test_probs)(significance_level=0.1)


        .. note::
            Update distance measure and penalty weights on-the-fly and predict again. The predictor will automatically
            apply the pending updates and recalibrate the scores before generating the new predictions.

            >>> icp.measure = 'norm'
            >>> icp.weight_hamming = 1.0
            >>> icp.weight_cardinality = 0.5
            >>> new_prediction_obj = icp.predict(test_probs)
            >>> new_prediction_sets = new_prediction_obj(significance_level=0.1)

        """

        if getattr(self, '_update_measure', False):
            if getattr(self, 'calib_probabilities', None) is not None and getattr(self, 'calib_labels',
                                                                                  None) is not None:
                self.calibrate()
            else:
                raise RuntimeError(
                    "Measure changed but no calibration data is cached. Please call calibrate() manually.")

        elif self._update_weight_hamming or self._update_weight_cardinality:
            self._update_calibration_scores()

        if self.sorted_calibration_scores is None:
            raise RuntimeError("Model is not calibrated.")

        probabilities = _is_tensor(probabilities).to(self.device)
        if probabilities.shape[1] != self.n_classes:
            raise RuntimeError("Test set probabilities must have the same number of columns as the number of classes.")

        cal_scores_ascending = torch.flip(self.sorted_calibration_scores.to(self.device), dims=[0])
        n_cal = len(cal_scores_ascending)

        max_combs = constants._GPU_MAX_COMBINATIONS if str(self.device).startswith('cuda') else constants._CPU_MAX_COMBINATIONS
        batch_size = max(1, max_combs // self.combinations.shape[0])

        p_values_list = []
        for i in tqdm(range(0, len(probabilities), batch_size), desc="Predicting"):
            batch_probs = probabilities[i: i + batch_size]
            batch_scores = self.all_combinations_scoring(batch_probs)

            flat_scores = batch_scores.view(-1)
            indices = torch.searchsorted(cal_scores_ascending, flat_scores, side='left')

            indices.neg_()
            indices.add_(n_cal + 1)

            batch_p_values_flat = indices.float()
            batch_p_values_flat.div_(n_cal + 1)
            batch_p_values = batch_p_values_flat.view(batch_scores.shape)
            p_values_list.append(batch_p_values.cpu().clone())

            del batch_scores, flat_scores, indices, batch_p_values_flat, batch_p_values

        final_p_values = torch.cat(p_values_list, dim=0)

        if torch.cuda.is_available() and constants._EMPTY_CUDA_CACHE:
            torch.cuda.empty_cache()

        return PredictionRegions(final_p_values, self.combinations, non_empty_prediction_regions)

    __call__ = predict