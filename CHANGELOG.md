# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-08-10

### Added
- `non_empty_prediction_regions` as a settable property on `PredictionRegions`
  (previously only configurable as a `predict()` argument), controlling
  whether empty prediction sets are corrected by including the label-set
  with the highest p-value.
- Observed fuzziness and observed excess efficiency metrics in
  `PredictionRegions.evaluate()`.
- Ledoit-Wolf shrinkage for the covariance matrix in
  `InductiveConformalPredictor`, ensuring a positive-definite distance
  matrix for the Mahalanobis measure.

### Changed
- Replaced `print()` calls with the standard `logging` module across
  `icp_predictor.py`, `icp_wrapper.py`, and `prediction_regions.py`.
  Lifecycle and state-change events log at `INFO`; internal tensor/shape
  diagnostics log at `DEBUG`. Consumers can now control verbosity instead
  of getting unconditional stdout output.
- `InductiveConformalPredictor.predict()` no longer accepts
  `non_empty_prediction_regions` as an argument; set it on the returned
  `PredictionRegions` object instead.
- Refactored `PredictionRegions` internals (metric calculation methods).

### Removed
- KS-test based validity check, superseded by the observed fuzziness/excess
  metrics above.

## [0.2.0] - 2026-04-08

### Added
- `constants.py` module exposing tunable memory/batching thresholds
  (`_CPU_MAX_COMBINATIONS`, `_GPU_MAX_COMBINATIONS`, `_REGION_BATCH_SIZE`,
  `_EMPTY_CUDA_CACHE`) so users can tune batching for their CPU/GPU memory
  limits during powerset scoring.

### Changed
- Major internal refactor of `icp_predictor.py`, `icp_wrapper.py`, and
  `prediction_regions.py` to batch heavy tensor operations and avoid
  Out-Of-Memory errors when scoring the full O(2^C) label powerset.
- Refined on-the-fly distance-measure and penalty-weight update behavior
  (recalibration without re-supplying calibration data).

## [0.1.0] - 2026-02-20

### Added
- Initial release: `ICPWrapper`, `InductiveConformalPredictor`, and
  `PredictionRegions` implementing Inductive Conformal Prediction for
  multi-label classification with Hamming and Cardinality structural
  penalties, supporting Mahalanobis and Euclidean Norm distance measures.
