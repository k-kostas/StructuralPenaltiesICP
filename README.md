# MuLaConf : Multi-Label Conformal Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/bsd-3-clause)

A flexible Python package for **Conformal Prediction (CP)** in **Multi-label** classification settings.
It implements the **Powerset Scoring** approach [[3]](#papadopoulos2014) using both the **Mahalanobis** 
distance [[1]](#katsios2024) and the standard **Euclidean Norm** [[4]](#maltou2022) as nonconformity measures, and applies
**Structural Penalties** to provide more informative prediction sets, based on Hamming distance
and label-set cardinality [[2]](#katsios2025). Designed for efficiency, it handles model training, calibration, 
and the on-the-fly update of structural penalty weights or distance measures without the need
for model retraining. This package bridges **Scikit-Learn** (for the underlying classifiers)
and **PyTorch** (for efficient tensor computations and GPU acceleration).

Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Quickstart](#quickstart)
- [Alternative Usage](#alternative-usage)
- [Examples](#examples)
- [Citing Structural Penalties ICP](#citing-structural-penalties-icp)
- [References](#references)


## Key Features

* **Multi-label Conformal Prediction**: Provides sets of label-sets with guaranteed coverage under the assumption of data exchangeability.
* **Powerset Scoring**: Explicitly assigns p-values to all possible label-sets.
* **Distance Measures**: Supports both the **Mahalanobis** distance and the standard **Euclidean Norm** in the error vector space.
* **Mahalanobis Nonconformity Measure**: Utilizes the Mahalanobis distance in the error vectors space to account for label correlations.
* **Structural Penalties**: Incorporates Hamming and Cardinality penalties to produce more informative prediction sets.
* **Post-training Penalty Updates**: Modify penalty weights or switch distance measures after fitting, without
retraining the underlying classifier. The package recalculates only the conformal components required by the updated configuration.
* **Automatic Classifier Switching**: Replace the underlying classifier (e.g., from `RandomForestClassifier` to `KNeighborsClassifier`) and the wrapper handles retraining automatically.
* **Compatible with any model**: Provides a wrapper (ICPWrapper) for any sklearn multi-label classifier (e.g., `MultiOutputClassifier`, `ClassifierChain`) plus a model agnostic InductiveConformalPredictor.
* **GPU Support**: Offloads heavy matrix computations to CUDA devices.


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

## Documentation
For the complete documentation see [mulaconf.readthedocs.io](https://mulaconf.readthedocs.io/en/latest/)


## Quickstart
This guide demonstrates the core usage of the MuLaConf package for a multi-label classification task 
to produce prediction sets for a new test sample in different significance levels. 

We will load the data, split it into proper training, calibration, and test sets, train the base model,
and evaluate the conformal predictions. We will use the Yeast dataset as an example, which has been preprocessed
into features and labels in CSV format. Note that the labels are represented as multi-hot vectors. 
The following code snippets run directly if you have cloned the package repository.

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

We initialize the underlying classifier from Scikit-Learn before fitting it on the proper training data. We have
chosen the `RandomForestClassifier` here, wrapped by `MultiOutputClassifier`. Then, we initialize the ICPWrapper
setting the model, the distance measure (default is `mahalanobis`) and the weights of the structural penalties
(default values are `0.0`). Notice that there are two ways to adjust the classifiers' arguments 
either by passing them directly

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from mulaconf.icp_wrapper import ICPWrapper

base_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=10))
wrapper = ICPWrapper(base_model, measure='mahalanobis', weight_hamming=2.0, weight_cardinality=1.5, device='cpu')
wrapper.fit(X_train, y_train)
```

or as a dictionary.   

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from mulaconf.icp_wrapper import ICPWrapper

base_model = MultiOutputClassifier(RandomForestClassifier())
wrapper = ICPWrapper(base_model, measure='mahalanobis', weight_hamming=2.0, weight_cardinality=1.5, device='cpu')
args = {'estimator__n_estimators': 10}
wrapper.fit(X_train, y_train, **args)
```

Once the model is fitted, the next step is calibration. This process uses the calibration set to compute
nonconformity scores, which are essential for calculating the p-values required to produce valid prediction regions.

```python
wrapper.calibrate(X_calib, y_calib)
```

> [!NOTE]
> **Switching Underlying Scikit-Learn Strategies** :
> You can switch the classification strategy or update its parameters. If the wrapper detects a change (via fingerprinting) during calibration, it will automatically retrain the new model on the cached proper training data.
>
> ```python
> from sklearn.neighbors import KNeighborsClassifier
> from sklearn.multioutput import ClassifierChain
>
> # Switch strategy to Classifier Chains with KNN
> wrapper.strategy = ClassifierChain(KNeighborsClassifier(n_neighbors=5))
>
> # Trigger automatic retraining and calibration
> wrapper.calibrate(X_calib, y_calib)
> ```


> [!NOTE]
> **On-the-fly Updates**: You can easily update the distance measure and penalty weights after the calibration
> process without passing your data again. Calling `calibrate()` without arguments will automatically
> apply all pending updates simultaneously using the cached calibration data.
>
> ```python
> wrapper.measure = 'norm'
> wrapper.weight_hamming = 1.0
> wrapper.weight_cardinality = 0.5
> wrapper.calibrate()
> ```



Finally, we generate prediction regions for the test set using the predict method. You can query this object to
extract valid label sets at a specific significance level
(e.g., $\alpha=0.1$ for 90% confidence) or multiple levels (e.g., $\alpha=[0.05, 0.1, 0.2]$).
The label-sets are returned as multi-hot vectors. In the example below, we retrieve the valid label combinations
for the first sample in the test set.


```python
prediction_obj = wrapper.predict(X_test)
prediction_regions = prediction_obj(significance_level=0.1)
print(prediction_regions[0])
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
prediction_regions = wrapper.predict(X_test)(significance_level=0.1)
```
> [!NOTE] 
> **Optional Non-empty Prediction Regions**: The `predict` method returns a `PredictionRegions` container holding the p-values and all candidate
> label-set combinations for each test sample. By default, this container ensures non-empty prediction regions
> by including the label-set with the highest p-value whenever no label-set satisfies the selected significance threshold.
> This behavior is controlled through the `non_empty_prediction_regions` property of the returned
> `PredictionRegions` object, which is set to `True` by default. Users can disable it by setting this
> property to `False` before extracting prediction regions.
>
>```python
>prediction_obj.non_empty_prediction_regions = False
>prediction_regions = prediction_obj(significance_level=0.1)
>```


> [!NOTE]  
> **On-the-Fly Updates**:
> You can update the distance measure and structural penalty weights after the prediction process
> The predictor automatically reconstructs the covariance matrix and recalibrates the conformal scores
> internally. You do not need to retrain the underlying classifier or pass your calibration data again.
> Simply assign the new parameters and call `predict()` again.
> 
> ```python
> # Update parameters directly via the wrapper
> wrapper.measure = 'norm'
> wrapper.weight_hamming = 1.0
> wrapper.weight_cardinality = 0.5
> 
> # Recalibration happens automatically on the fly, so you can call the predict() method immediately.
> updated_prediction_obj = wrapper.predict(X_test)
> updated_prediction_regions = updated_prediction_obj(significance_level=0.1)
> ```


> [!NOTE]
> **Accessing P-Values**: You also have direct access to the raw p-values for every possible label combination.
> Below, we print the p-values for the first test sample.
>
> ```python
> print(prediction_obj.p_values[0])
>```
>
> ```text
> tensor([0.0627, 0.0015, 0.0719,  ..., 0.0015, 0.0015, 0.0015])
> ```


The `evaluate` method provides a convenient way to calculate performance metrics, including Coverage, 
N-Criterion, S-Criterion, Observed Fuzziness and Observed Excess. Additionally, it can return the p-values
corresponding to the true labels.

The method requires the **ground truth labels** (`true_labelsets`) and the desired **significance level**.
All other metric-specific arguments are optional boolean flags, which default to `True` if not specified.

```python
metrics = prediction_obj.evaluate(
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
You can also use the InductiveConformalPredictor class as a standalone package if you prefer to manage the underlying
classifier yourself or use a framework other than Scikit-Learn. In this mode, you must provide the **predicted probabilities** for the
proper training, calibration, and test sets, as well as the **ground truth label-sets** for the training and calibration sets.

The package is flexible regarding input formats: it accepts PyTorch Tensors, NumPy arrays, Pandas DataFrames/Series,
or lists. All data is automatically converted to tensors and moved to the specified device (CPU or GPU) for 
efficient processing.

First, we need to initialize the InductiveConformalPredictor class to calculate the structural penalties and to form
the covariance matrix using the error vectors.

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

Next, we call the `calibrate` method to calculate the calibration scores based on the calibration probabilities
and labels.

```python
icp.calibrate(probabilities=calib_probs,labels=calib_labels)
```
Then, we can generate prediction regions for the test set by calling the `predict` method and passing the test
probabilities. You can query this object to
extract valid label-sets at a specific significance level
(e.g., $\alpha=0.1$ for 90% confidence) or multiple levels (e.g., $\alpha=[0.05, 0.1, 0.2]$).
The label-sets are returned as multi-hot vectors. In the example below, we retrieve the valid label combinations
for the first sample in the test set.


```python
prediction_obj = icp.predict(test_probs)
prediction_regions = prediction_obj(significance_level=0.1)
print(prediction_regions[0])
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
prediction_regions = icp.predict(test_probs)(significance_level=0.1)
```

And of course, we have access to the p-values. In the example below, we get the p-values of the first sample in the
test set.

```python
print(prediction_obj.p_values[0])
```

```text
tensor([0.0627, 0.0015, 0.0719,  ..., 0.0015, 0.0015, 0.0015])
```


> [!NOTE] 
> **Optional Non-empty Prediction Regions**: The `predict` method returns a `PredictionRegions` container holding the p-values and all candidate
> label-set combinations for each test sample. By default, this container ensures non-empty prediction regions
> by including the label-set with the highest p-value whenever no label-set satisfies the selected significance threshold.
> This behavior is controlled through the `non_empty_prediction_regions` property of the returned
> `PredictionRegions` object, which is set to `True` by default. Users can disable it by setting this
> property to `False` before extracting prediction regions.
>
> ```python
> prediction_obj.non_empty_prediction_regions = False
> prediction_regions = prediction_obj(significance_level=0.1)
> ```


Also, it allows us to get the p-values of the true test set label-sets and evaluate metrics like Coverage, N-Criterion,
S-Criterion, Observed Fuzziness and Observed Excess. 

```python
metrics = prediction_obj.evaluate(
    return_true_label_p_value = False,
    return_coverage=True,
    return_n_criterion=True,
    return_s_criterion=True,
    return_observed_fuzziness=True,
    return_observed_excess=True,
    true_labelsets=test_labels,
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

> [!NOTE]  
> **On-the-Fly Updates**: 
> You can update the distance measure and structural penalty weights
> at any time. The predictor automatically reconstructs the covariance matrix and recalibrates the conformal scores
> internally. You do not need to retrain the underlying classifier or pass your calibration data again.
> Assign the new parameters and call `predict()` again.
> 
> ```python
> # Update parameters directly
> icp.measure = 'norm'
> icp.weight_hamming = 2.0
> icp.weight_cardinality = 1.5
> 
> # Recalibration happens automatically on the fly, so you can call the predict() method immediately.
> updated_prediction_obj = icp.predict(test_probs)
> updated_prediction_regions = updated_prediction_obj(significance_level=0.1)
> ```


## Examples

For additional examples of how to use the package, see the [documentation](https://mulaconf.readthedocs.io/en/latest/documentation.html).


## Citing MuLaConf

If you use the package for a scientific publication, you are kindly requested to cite the following paper:

> <a id="katsios2025"></a>Katsios, K., & Papadopoulos, H. (2025). Incorporating Structural Penalties in Multi-label Conformal Prediction.
> *Proceedings of Machine Learning Research*, 266, 1-20.
[[Download PDF](https://proceedings.mlr.press/v230/katsios24a.html)]

**BibTeX:**

```bibtex
@article{katsios2025incorporating,
  title={Incorporating Structural Penalties in Multi-label Conformal Prediction},
  author={Katsios, Kostas and Papadopoulos, Harris},
  journal={Proceedings of Machine Learning Research},
  volume={266},
  pages={1--20},
  year={2025}
}
```


## References

1. <a id="katsios2025"></a>Katsios, K., & Papadopoulos, H. (2025). Incorporating Structural Penalties in Multi-label Conformal Prediction.
    *Proceedings of Machine Learning Research*, 266, 1-20. [Proceedings](https://proceedings.mlr.press/v266/katsios25a.html)

2. <a id="katsios2024"></a>Katsios, K., & Papadopoulos, H. (2024). Multi-label conformal prediction with a Mahalanobis distance nonconformity measure.
    *Proceedings of Machine Learning Research*, 230, 1-14. [Proceedings](https://proceedings.mlr.press/v230/katsios24a.html)

3. <a id="papadopoulos2014"></a>Papadopoulos, H. (2014). A cross-conformal predictor for multi-label classification. In *Artificial Intelligence Applications and Innovations: AIAI 2014 Workshops: CoPA, MHDW, IIVC, and MT4BD, Rhodes, Greece, September 19-21, 2014. Proceedings 10* (pp. 241–250). Springer. [DOI: 10.1007/978-3-662-44722-2_26](https://doi.org/10.1007/978-3-662-44722-2_26)

4. <a id="maltou2022"></a>Maltoudoglou, L., Paisios, A., Lenc, L., Martı́nek, J., Král, P., & Papadopoulos, H. (2022). Well-calibrated confidence measures for multi-label text classification with a large number of labels. *Pattern Recognition*, 122, 108271. [DOI: 10.1016/j.patcog.2021.108271](https://doi.org/10.1016/j.patcog.2021.108271)

5. <a id="lambrou2016"></a>Lambrou, A., & Papadopoulos, H. (2016). Binary relevance multi-label conformal predictor. In *Conformal and Probabilistic Prediction with Applications* (pp. 90–104). Springer. [DOI: 10.1007/978-3-319-33395-3_7](https://doi.org/10.1007/978-3-319-33395-3_7)

6. <a id="papadopoulos2002a"></a>Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002a). Inductive confidence machines for regression. In *Machine learning: ECML 2002: 13th European conference on machine learning Helsinki, Finland, August 19–23, 2002 proceedings 13* (pp. 345–356). Springer. [DOI: 10.1007/3-540-36755-1_29](https://doi.org/10.1007/3-540-36755-1_29)

7. <a id="papadopoulos2002b"></a>Papadopoulos, H., Vovk, V., & Gammerman, A. (2002b). Qualified prediction for large data sets in the case of pattern recognition. In *ICMLA* (pp. 159–163).

8. <a id="vovk2005"></a>Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World* (Vol. 29). Springer. [DOI: 10.1007/b106715](https://doi.org/10.1007/b106715)

9. <a id="vovk2016"></a>Vovk, V., Fedorova, V., Nouretdinov, I., & Gammerman, A. (2016). Criteria of efficiency for conformal prediction. In *Conformal and Probabilistic Prediction with Applications: 5th International Symposium, COPA 2016, Madrid, Spain, April 20-22, 2016, Proceedings 5* (pp. 23–39). Springer. [DOI: 10.1007/978-3-319-33395-3_2](https://doi.org/10.1007/978-3-319-33395-3_2)