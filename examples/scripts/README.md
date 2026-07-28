# Examples: Scripts

This folder contains runnable experiment scripts for benchmarking and reproducing MuLaConf results.

## Available Script

- `running_experiments_presenting_results.py`  
  Runs cross-validation experiments, compares CPU/GPU execution, and exports timing/metric outputs, tables, and plots.

## Quick Start

1. Open the script and set the `dataset` name (e.g., `yeast` or `tmc2007_500`). 
    Also verify: `calibration_size = 0.3`.
2. The experiment uses **Repeated K-Fold cross-validation** with:
- `n_splits = 10`
- `n_repeats = 1`
\
So the default setup is **10-fold cross-validation** (10 total iterations).
3. From the project root, run.

    ```bash
    python examples/scripts/running_experiments_presenting_results.py
    ```

## Notes

- Requires Python 3.10+ and project dependencies installed.
- If CUDA is available, the script benchmarks both CPU and GPU.


## What the script does

- Downloads X_<dataset>.csv and y_<dataset>.csv
- Creates train/calibration/test splits within repeated CV
Effective split sizes per fold:
  - **Test size:** `1 / n_splits = 10%` of the full dataset
  - Remaining **90%** is used as temporary training data, then split into:
    - **Calibration size:** `calibration_size = 0.3` of that 90% (i.e., **27%** of full dataset)
    - **Proper training size:** remaining **63%** of full dataset
- Trains a base multi-label model (XGBoost via MultiOutputClassifier)
- Benchmarks conformal prediction for:
  - measures: mahalanobis, norm
  - weight settings:
    - W_H=0, W_C=0
    - W_H=1, W_C=1
    - W_H=0, W_C=1
    - W_H=1, W_C=0
- Computes coverage and prediction-region size across alpha values
- Saves CSV summaries and paper-ready artifacts (LaTeX + plots)


## Outputs

The script creates an output folder named after the selected dataset and stores:

- Raw timing CSVs
- Timing summary CSVs
- Empirical validity and prediction-size CSVs
- LaTeX tables (for paper/report use)
- PNG plots


## Output location

Results are written to:

<dataset>_experiments_testing/

Expected outputs include:
- raw_timing_default_<dataset>.csv
- raw_timing_penalty_<dataset>.csv
- timing_default_case_<dataset>.csv
- timing_penalty_cases_<dataset>.csv
- empirical_validity_<dataset>.csv
- prediction_sizes_<dataset>.csv

And inside:
<dataset>_experiments_testing/paper_data/
- predidiction_sizes_percentages.tex
- penalty_avg_timing.tex
- default_avg_timing_per_device.tex
- plot_prediction_sizes_<dataset>_paper.png
- plot_empirical_validity_<dataset>.png

## Notes

- If CUDA is not available, the script runs CPU only.
- The script sets:
  constants._EMPTY_CUDA_CACHE = False
  for benchmark behavior.
- Runtime can be long due to cross-validation and many alpha values.
- Ensure enough RAM/VRAM for large label spaces.



