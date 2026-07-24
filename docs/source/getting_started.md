# Getting Started

## Installation
MuLaConf is available on both PyPI and Conda-forge.

From PyPI:
```bash
pip install mulaconf
```

From Conda-forge:
```bash
conda install -c conda-forge mulaconf
```


## Tutorial
MuLaConf is a Python package implementing the novel Conformal Prediction framework introduced
in "Incorporating Structural Penalties in Multi-label Conformal Prediction" 
[[Katsios & Papadopoulos, 2025](https://proceedings.mlr.press/v266/katsios25a.html)]. 
Based on this research, the package integrates distance metrics and structural penalties. By penalizing the
nonconformity scores based on the label dependencies of the training data, it produces valid,
informative prediction sets.

The package offers configurations across two main components:

**1. Distance Measures**

Used to calculate the base nonconformity scores:
* **Mahalanobis:** Accounts for label correlations by forming the covariance matrix of the error vectors.
* **Euclidean:** A standard, unweighted baseline distance metric.

**2. Structural Penalties**

Can be applied on top of the distance measures to further reduce the size of the prediction sets:
* **Hamming Penalty:** Penalizes label combinations based on their minimum Hamming distance from the true label-sets
found in the proper-training data.
* **Cardinality Penalty:** Penalizes label combinations whose size (number of active labels) deviates significantly
from the expected cardinality of the proper-training set.


### Load and split data
We will load the data, split it into proper-training, calibration and test sets, train the model and evaluate the conformal
predictions. For example, we will use the **Yeast** dataset after we have preprocessed the data into features and labels
in CSV format.

~~~{Note}
The labels should be represented as **multi-hot vectors**.
~~~

~~~{Note}
The example works directly when the package is cloned.
~~~

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Define the path to your data
data_path = "/data/yeast"

# 2. Load the Yeast dataset (Features and Labels)
X = pd.read_csv(f"{data_path}/X_yeast.csv")
y = pd.read_csv(f"{data_path}/y_yeast.csv")

# 3. Split the data
# First, separate out the Test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Then, split the remaining data into Proper Train and Calibration (30%)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
```

```text
Loading Yeast dataset...
Data shapes: Train=(1522, 103), Calib=(653, 103), Test=(242, 103)
```

### Setting up the Wrapper - Scikit-Learn API
The ICPWrapper is a Scikit-Learn compatible interface that integrates MuLaConf with standard
Scikit-Learn classifiers. It bridges Scikit-Learn (for the underlying classifier) and PyTorch
(for efficient tensor computations and GPU acceleration).

To initialize the wrapper, you must provide the base classifier and, optionally, the structural penalty weights.

  * Penalty Weights: The `weight_hamming` and `weight_cardinality` parameters control the strength of the penalties.
    They must be non-negative real values (default is 0.0).

  * Device: The device parameter specifies where tensor computations occur. Set this to `'cpu'` (default)
    or `'cuda'` to leverage GPU acceleration.

  ~~~{Note}
   **GPU Acceleration**:
     The underlying Scikit-Learn classifier always runs on the CPU. The device parameter only affects 
     the Conformal Prediction and Structural Penalty calculations.
   ~~~


#### Handling Classifier Arguments and Fitting
The fit method of the ICPWrapper accepts the features and the multi-hot vectors for the true label-sets of the 
proper-training data. It trains the underlying classifier, calculates the covariance matrix and the structural
penalties for the inductive conformal predictor.

When using meta-estimators (e.g., [`MultiOutputClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html),
[`ClassifierChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html),
[`OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.OneVsRestClassifier.html)),
you can configure the base learnern's parameters in two ways:

1. Direct Initialization
   Initialize the inner classifier with its parameters before passing it to the wrapper.

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    from mulaconf.icp_wrapper import ICPWrapper
    
    base_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=10))
    wrapper = ICPWrapper(base_model, weight_hamming=2.0, weight_cardinality=1.5, device='cpu')
    wrapper.fit(X_train, y_train)
    ```

2. You can pass parameters to the fit method using a dictionary. For meta-estimators, you must use the
    `estimator__` prefix (double underscore) to reach the inner model.

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    
    from mulaconf.icp_wrapper import ICPWrapper
    
    base_model = MultiOutputClassifier(RandomForestClassifier())
    wrapper = ICPWrapper(base_model, weight_hamming=2.0, weight_cardinality=1.5, device='cpu')
    args = {'estimator__n_estimators': 5}
    wrapper.fit(X_train, y_train, **args)
    ```

    ~~~{Note}
    **Native Estimators:** If you are using a standalone multi-label classifier
    (e.g., [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    or [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    directly), you do not use the `estimator__` prefix. You can pass arguments directly (e.g., `n_estimators=10` or `{'n_estimators': 10}`).
    ~~~


### Calibration
Once the model is fitted, the next step is calibration. This process uses the calibration set to compute
nonconformity scores, which are essential for calculating the p-values required to produce valid prediction regions.

```python
wrapper.calibrate(X_calib, y_calib)
```


~~~{Note}
**Switching Underlying Scikit-Learn Strategies** :
 You can switch the classification strategy or update its parameters. If the wrapper detects a change (via fingerprinting) during calibration, it will automatically retrain the new model on the cached proper training data.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain

# Switch strategy to Classifier Chains with KNN
wrapper.strategy = ClassifierChain(KNeighborsClassifier())
wrapper.kwargs = {'estimator__n_neighbors': 5}

# Trigger automatic retraining and calibration
wrapper.calibrate(X_calib, y_calib)
```
~~~

~~~{Note}
**On-the-fly Updates**: You can easily update the distance measure and penalty weights after the calibration
process without passing your data again. Calling `calibrate()` without arguments will automatically
apply all pending updates simultaneously using the cached calibration data.

```python
wrapper.measure = 'norm'
wrapper.weight_hamming = 1.0
wrapper.weight_cardinality = 0.5
wrapper.calibrate()
```
~~~


### Predicting Valid Conformal Regions and Evaluation Metrics

#### Prediction
Finally, we generate prediction regions for the test set using the predict method.

```python
prediction_regions_obj = wrapper.predict(X_test)
```

The predict method returns a PredictionRegions container holding the conformal prediction regions for each sample.
By default, the predictor guarantees non-empty prediction sets by always including the label-set with the highest p-value.
You can switch this behavior using the `non_empty_prediction_regions` boolean parameter (default `True`).
You can query this object to extract valid label sets at a specific significance level
(e.g., $\alpha=0.1$ for 90% confidence) or multiple levels (e.g., $\alpha=[0.05, 0.1, 0.2]$).

The label-sets are returned as multi-hot vectors. In the example below, we retrieve the valid label combinations
for the first sample in the test set.

```python
prediction_sets = prediction_regions_obj(significance_level=0.1)
print(prediction_sets[0])
```

```text
tensor([[0, 0, 0,  ..., 1, 1, 0],
        [0, 0, 0,  ..., 1, 0, 0],
        [0, 0, 0,  ..., 1, 1, 0],
        ...,
        [1, 1, 1,  ..., 1, 1, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 0]], dtype=torch.int32)
```

Equivalent one-liner:

```python
prediction_sets = wrapper.predict(X_test)(significance_level=0.1)
```

~~~{Note}
**On-the-fly Updates**: Update diastance measure and penalty weights on-the-fly and predict again.
The predictor will automatically apply the pending updates and recalibrate the scores before generating
the new predictions.

```python
wrapper.measure = 'norm'
wrapper.weight_hamming = 1.0
wrapper.weight_cardinality = 0.5
new_prediction_obj = wrapper.predict(X_test)
new_prediction_sets = new_prediction_obj(significance_level=0.1)
```
~~~

~~~{Note}
**Accessing P-Values**: You also have direct access to the raw p-values for every possible label combination.
Below, we print the p-values for the first test sample.

```python
print(prediction_regions_obj.p_values[0])
```

```text
tensor([0.0627, 0.0015, 0.0719,  ..., 0.0015, 0.0015, 0.0015])
```
~~~

#### Evaluation
The `evaluate` method provides a convenient way to calculate performance metrics, including Coverage, 
N-Criterion, S-Criterion, Observed Fuzziness and Observed Excess. Additionally, it can return the p-values
corresponding to the true labels.

##### Arguments
The method requires the **ground truth labels** ('true_labelsets') and the desired **significance level**.
All other metric-specific arguments are optional boolean flags, which default to `True` if not specified.

```python
metrics = prediction_regions_obj.evaluate(
    return_true_label_p_value = False,
    return_coverage=True,
    return_n_criterion=True,
    return_s_criterion=True,
    return_observed_fuzziness=True,
    return_observed_excess=True,
    true_labelsets=y_test,
    significance_level=0.1,
)

print(metrics)
```

```text
{
    'coverage': 0.9008264541625977,
    'n_criterion': 957.6694214876034,
    'observed_excess': 956.7685950413223,
    's_criterion': 457.9545593261719,
    'observed_fuzziness': 457.46160888671875
}
```

## Alternative usage
You can also use the InductiveConformalPredictor class as a standalone engine if you prefer to manage the underlying
classifier yourself or not using Scikit-Learn. In this mode, you must provide the **predicted probabilities** for the
proper training, calibration, and test sets, as well as the **ground truth labels** for the training and calibration sets.

The package is flexible regarding input formats: it accepts PyTorch Tensors, NumPy arrays, Pandas DataFrames/Series,
or lists. All data is automatically converted to tensors and moved to the specified device (CPU or GPU) for 
efficient processing.

1. ### Initialization
First, we need to initialize the InductiveConformalPredictor class to calculate the structural penalties and to form
the covariance matrix using the proper training data.

```python
from mulaconf.icp_predictor import InductiveConformalPredictor

icp = InductiveConformalPredictor(
    predicted_probabilities=train_probs,
    true_labels=train_labels,
    measure='mahalanobis',
    weight_hamming=1.5,
    weight_cardinality=0.5,
    device='cpu'
)
```

2. ### Calibration
Next, we call the `calibrate` method to calculate the calibration scores based on the calibration probabilities
and labels.

```python
icp.calibrate(probabilities=calib_probs,labels=calib_labels)
```


~~~{Note}
**On-the-fly Updates**: You can update the distance `measure` ('norm' or 'mahalanobis'), `weight_hamming`,
and `weight_cardinality` at any time after the calibration process.

The predictor utilizes lazy evaluation for automatic recalibration. This means
you do not need to manually pass your calibration data again or explicitly call
`calibrate()`. Simply assign new values to the properties (e.g., `icp.measure = 'norm'`,
`icp.weight_hamming = 2.0`, `icp.weight_cardinality = 1.5`) and immediately call `calibrate()`. It will
automatically reform the underlying covariance matrix and recalibrate the scores
on-the-fly.

1. The `calibrate` method recalculates the distance matrix and scores after a measure update.
```python
icp.measure = 'norm'
icp.calibrate()
```

2. The `calibrate` method recalculates calibration scores after penalty weight update.
```python
icp.weight_hamming = 2.0
icp.weight_cardinality = 1.5
icp.calibrate()
```

3. The `calibrate` method recalculates the covariance matrix and nonconformity scores after updating measure
and penalty weights.
```python 
icp.measure = 'norm'
icp.weight_hamming = 2.0
icp.weight_cardinality = 1.5
icp.calibrate()
```
~~~


3. ### Predict Valid Conformal Regions
We generate prediction regions for the test set using the predict method.

```python
prediction_regions_obj = icp.predict(test_probs)
```

The predict method returns a PredictionRegions container holding the conformal prediction regions for each sample.
By default, the predictor guarantees non-empty prediction sets by always including the label-set with the highest p-value.
You can switch this behavior using the `non_empty_prediction_regions` boolean parameter (default `True`).
You can query this object to extract valid label sets at a specific significance level
(e.g., $\alpha=0.1$ for 90% confidence) or multiple levels (e.g., $\alpha=[0.05, 0.1, 0.2]$).

The label-sets are returned as multi-hot vectors. In the example below, we retrieve the valid label combinations
for the first sample in the test set.

```python
prediction_sets = prediction_regions_obj(significance_level=0.1)
print(prediction_sets[0])
```

```text
tensor([[0, 0, 0,  ..., 1, 1, 0],
        [0, 0, 0,  ..., 1, 0, 0],
        [0, 0, 0,  ..., 1, 1, 0],
        ...,
        [1, 1, 1,  ..., 1, 1, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 0]], dtype=torch.int32)
```

Equivalent one-liner:

```python
prediction_sets = icp.predict(test_probs)(significance_level=0.1)
```

~~~{Note}
**On-the-fly Updates**: Update diastance measure and penalty weights on-the-fly and predict again.
The predictor will automatically apply the pending updates and recalibrate the scores before generating
the new predictions.

```python
icp.measure = 'norm'
icp.weight_hamming = 2.0
icp.weight_cardinality = 1.5
new_prediction_obj = icp.predict(test_probs)
new_prediction_sets = new_prediction_obj(significance_level=0.1)
```
~~~


~~~{Note}
**Accessing P-Values**: You also have direct access to the raw p-values for every possible label combination.
Below, we print the p-values for the first test sample.

```python
print(prediction_regions_obj.p_values[0])
```

```text
tensor([0.0627, 0.0015, 0.0719,  ..., 0.0015, 0.0015, 0.0015])
```
~~~

4. ### Evaluation Metrics
The `evaluate` method provides a convenient way to calculate performance metrics, including Coverage, 
N-Criterion, S-Criterion, Observed Fuzziness and Observed Excess. Additionally, it can return the p-values
corresponding to the true labels.

**Arguments**


The method requires the **ground truth labels** ('true_labelsets') and the desired **significance level**.
All other metric-specific arguments are optional boolean flags, which default to `True` if not specified.

```python
metrics = prediction_regions_obj.evaluate(
    return_true_label_p_value = False,
    return_coverage=True,
    return_n_criterion=True,
    return_s_criterion=True,
    return_observed_fuzziness=True,
    return_observed_excess=True,
    true_labelsets=y_test,
    significance_level=0.1,
)

print(metrics)
```

```text
{
    'coverage': 0.9008264541625977,
    'n_criterion': 957.6694214876034,
    'observed_excess': 956.7685950413223,
    's_criterion': 457.9545593261719,
    'observed_fuzziness': 457.46160888671875
}
```


## Performance & Memory Management

Because Powerset Scoring evaluates all possible label combinations, the prediction math scales
exponentially at $O(2^C)$ where $C$ is the number of classes. 

To prevent Out-Of-Memory (OOM) crashes on standard hardware, **MuLaConf** uses batching and tensor expansion.
The default limits are tuned for standard laptops (8GB RAM) and standard GPUs (8GB VRAM). 

If you are running on high-end hardware, or if you are conducting rigorous execution-time benchmarks,
you can manually override the module-level constants to dramatically speed up execution.

### How to tune the hardware limits
Simply import the `constants` module and overwrite the values before initializing or running the predictor:

```python
import mulaconf.constants as constants
from mulaconf.icp_wrapper import ICPWrapper

# 1. Increase GPU batching for high-end cards (e.g., 24GB VRAM)
constants._GPU_MAX_COMBINATIONS = 15_000_000

# 2. Turn off cache-clearing for raw benchmarking
# (Removes the synchronization delay, but risks VRAM fragmentation)
constants._EMPTY_CUDA_CACHE = False

# 3. Run your code normally
wrapper = ICPWrapper(base_model)
wrapper.fit(X_train, y_train)
```

~~~{Note}
**Hardware Cheat Sheet**:
For a comprehensive guide on exactly what batching numbers to use for your specific RAM and GPU VRAM setup,
refer to the documentation at the top of the `constants.py` file.
~~~