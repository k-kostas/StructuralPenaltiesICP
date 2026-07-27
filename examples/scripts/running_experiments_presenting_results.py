import pandas as pd
import numpy as np
import torch
import time
import os
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests

from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from mulaconf.icp_predictor import InductiveConformalPredictor
import mulaconf.constants as constants

constants._EMPTY_CUDA_CACHE = False


# --- Setup  ---
# Dataset: "yeast" or "tmc2007_500"
dataset = ""
calibration_size = 0.3

# Download data from data folder on GitHub
data_dir = os.path.join(os.getcwd(), f"data_testing/{dataset}")
os.makedirs(data_dir, exist_ok=True)

base_url = f"https://raw.githubusercontent.com/k-kostas/MuLaConf/main/data/{dataset}"

x_output_path = os.path.join(data_dir, f"X_{dataset}.csv")
y_output_path = os.path.join(data_dir, f"y_{dataset}.csv")

# Download X data
print(f"Downloading X_{dataset}.csv...")
response_x = requests.get(f"{base_url}/X_{dataset}.csv", allow_redirects=True, timeout=60)
response_x.raise_for_status()
with open(x_output_path, "wb") as f:
    f.write(response_x.content)

# Download y data
print(f"Downloading y_{dataset}.csv...")
response_y = requests.get(f"{base_url}/y_{dataset}.csv", allow_redirects=True, timeout=60)
response_y.raise_for_status()
with open(y_output_path, "wb") as f:
    f.write(response_y.content)

out_dir = f"{dataset}_experiments_testing"
os.makedirs(out_dir, exist_ok=True)

print(f"Loading {dataset} dataset...")
X = pd.read_csv(x_output_path).values
y = pd.read_csv(y_output_path).values
print(f"X shape: {X.shape}, y shape: {y.shape}")

classes = y.shape[1]

# --- Check devices ---
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')
    print("CUDA detected! Will run a fair CPU vs GPU benchmark.")
else:
    print("No CUDA detected. Running CPU only.")


# --- Timing helper functon ---
def get_time(device):
    """Ensures GPU operations are finished before stopping the stopwatch."""
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return time.perf_counter()


# --- CUDA Warmup function ---
def warmup_cuda():
    """Performs CUDA warmup to avoid cold-start penalties on first operations."""
    if torch.cuda.is_available():
        print("\nWarming up CUDA context to prevent Fold 1 timing penalties...")
        for _ in range(3):
            _ = torch.rand(1000, 1000, device='cuda') @ torch.rand(1000, 1000, device='cuda')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if 'cuda' in devices:
    warmup_cuda()


# --- Format MultiOutputClassifier probabilities for MuLaConf ---
def get_positive_probs(proba_list):
    """Converts a list of (N, 2) arrays into an (N, C) matrix of positive class probabilities."""
    return np.column_stack([p[:, 1] for p in proba_list])

# --- Cleanup memory ---
def cleanup_memory(device):
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Experiment Parameters ---
n_splits = 10
n_repeats = 1
total_folds = n_splits * n_repeats
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

alphas = [round(a, 2) for a in np.arange(0.01, 1.00, 0.01)]
measures = ['mahalanobis', 'norm']

# Separate default case from penalty update cases
default_weight_case = ("W_H=0, W_C=0", (0.0, 0.0))
penalty_weight_cases = {
    "W_H=1, W_C=1": (1.0, 1.0),
    "W_H=0, W_C=1": (0.0, 1.0),
    "W_H=1, W_C=0": (1.0, 0.0)
}

# All weight cases for results storage
all_weight_cases = {default_weight_case[0]: default_weight_case[1], **penalty_weight_cases}

# Generate all 8 combinations (4 weights x 2 measures)
all_cases = [f"{m[:3].upper()} | {w}" for m in measures for w in all_weight_cases.keys()]
print(f"Total number of weight-case combinations: {len(all_cases)}")
print(f"Cases: {all_cases}")

# Data structures to hold results across all folds
coverage_results = {device: {case: np.zeros((total_folds, len(alphas))) for case in all_cases} for device in devices}
size_results = {device: {case: np.zeros((total_folds, len(alphas))) for case in all_cases} for device in devices}

# Timing records for default case (init, calibrate, predict, evaluate)
default_timing_records = []

# Timing records for penalty update cases (penalty_update, predict, evaluate)
penalty_timing_records = []

total_start_time = time.perf_counter()

# --- Main 10x10 CV Loop ---
print(f"\nStarting 10x10 Cross-Validation ({total_folds} total iterations)...")
fold_iterator = tqdm(enumerate(rkf.split(X, y)), total=total_folds, desc="CV Progress")

for iteration, (train_index, test_index) in fold_iterator:
    X_cv_train, X_test = X[train_index], X[test_index]
    y_cv_train, y_test = y[train_index], y[test_index]
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_cv_train, y_cv_train, test_size=calibration_size, random_state=iteration
    )

    # =====================================================================
    # STEP A: Train XGBoost Base Model ONCE per fold
    # =====================================================================
    base_model = MultiOutputClassifier(XGBClassifier(n_estimators=50, tree_method='hist', random_state=iteration))
    base_model.fit(X_train, y_train)

    train_probs = get_positive_probs(base_model.predict_proba(X_train))
    calib_probs = get_positive_probs(base_model.predict_proba(X_calib))
    test_probs = get_positive_probs(base_model.predict_proba(X_test))

    # =====================================================================
    # STEP B: Raw ICP Engine Benchmark (CPU vs GPU)
    # =====================================================================
    for device in devices:
        for measure in measures:
            cleanup_memory(device)

            # =============================================================
            # DEFAULT CASE: W_H=0, W_C=0 (Full timing: init, calibrate, predict, evaluate)
            # =============================================================
            default_name, (default_w_h, default_w_c) = default_weight_case
            case_key = f"{measure[:3].upper()} | {default_name}"

            default_timing = {
                'Iteration': iteration,
                'Device': device,
                'Measure': measure,
                'Weight_Case': default_name,
                'W_H': default_w_h,
                'W_C': default_w_c
            }

            # --- Init Time ---
            t0 = get_time(device)
            icp = InductiveConformalPredictor(
                predicted_probabilities=train_probs,
                true_labels=y_train,
                measure=measure,
                weight_hamming=default_w_h,
                weight_cardinality=default_w_c,
                device=device
            )
            default_timing['ICP_Init_Time'] = get_time(device) - t0

            # --- Calibrate Time ---
            t0 = get_time(device)
            icp.calibrate(calib_probs, y_calib)
            default_timing['ICP_Calibrate_Time'] = get_time(device) - t0

            # --- Predict Time ---
            t0 = get_time(device)
            prediction_regions = icp.predict(test_probs)
            default_timing['Predict_Time'] = get_time(device) - t0

            # --- Evaluate Time ---
            t0 = get_time(device)
            metrics = prediction_regions.evaluate(
                true_labelsets=y_test,
                significance_level=alphas,
                return_true_label_p_value=False,
                return_coverage=True,
                return_n_criterion=True,
                return_observed_fuzziness=False,
                return_observed_excess=False,
                return_s_criterion=False
            )
            default_timing['Evaluate_Time'] = get_time(device) - t0

            # --- Total time for default case ---
            default_timing['Case_Total_Time'] = (
                    default_timing['ICP_Init_Time'] +
                    default_timing['ICP_Calibrate_Time'] +
                    default_timing['Predict_Time'] +
                    default_timing['Evaluate_Time']
            )

            default_timing_records.append(default_timing)

            # Store coverage and size results for default case
            for a_idx, alpha in enumerate(alphas):
                coverage_results[device][case_key][iteration, a_idx] = metrics[alpha]['coverage']
                size_results[device][case_key][iteration, a_idx] = metrics[alpha]['n_criterion']

            del prediction_regions, metrics

            # =============================================================
            # PENALTY UPDATE CASES: Reuse calibrated ICP, only update weights
            # =============================================================
            for weight_name, (w_h, w_c) in penalty_weight_cases.items():
                case_key = f"{measure[:3].upper()} | {weight_name}"

                penalty_timing = {
                    'Iteration': iteration,
                    'Device': device,
                    'Measure': measure,
                    'Weight_Case': weight_name,
                    'W_H': w_h,
                    'W_C': w_c
                }

                # --- Penalty Update Time ---
                t0 = get_time(device)
                icp.weight_hamming = w_h
                icp.weight_cardinality = w_c

                # Force the engine to apply the new weights and re-sort the scores NOW!
                icp.calibrate()

                penalty_timing['Recalibration_Time'] = get_time(device) - t0

                # --- Predict Time ---
                t0 = get_time(device)
                prediction_regions = icp.predict(test_probs)
                penalty_timing['Predict_Time'] = get_time(device) - t0

                # --- Evaluate Time ---
                t0 = get_time(device)
                metrics = prediction_regions.evaluate(
                    true_labelsets=y_test,
                    significance_level=alphas,
                    return_true_label_p_value=False,
                    return_coverage=True,
                    return_n_criterion=True,
                    return_observed_fuzziness=False,
                    return_observed_excess=False,
                    return_s_criterion=False
                )
                penalty_timing['Evaluate_Time'] = get_time(device) - t0

                # --- Total time for penalty case ---
                penalty_timing['Case_Total_Time'] = (
                        penalty_timing['Recalibration_Time'] +
                        penalty_timing['Predict_Time'] +
                        penalty_timing['Evaluate_Time']
                )

                penalty_timing_records.append(penalty_timing)

                # Store coverage and size results
                for a_idx, alpha in enumerate(alphas):
                    coverage_results[device][case_key][iteration, a_idx] = metrics[alpha]['coverage']
                    size_results[device][case_key][iteration, a_idx] = metrics[alpha]['n_criterion']

                del prediction_regions, metrics

            # Cleanup ICP instance after all weight cases for this measure
            del icp
            cleanup_memory(device)

    # Aggressive memory cleanup per fold
    del base_model, train_probs, calib_probs, test_probs
    del X_train, X_calib, X_test, y_train, y_calib, y_test
    cleanup_memory('cuda' if 'cuda' in devices else 'cpu')

# --- Process and Save DataFrames ---
print("\nProcessing results and saving DataFrames...")

# Create timing DataFrames
df_default = pd.DataFrame(default_timing_records)
df_penalty = pd.DataFrame(penalty_timing_records)

# Summary for default case (init, calibrate, predict, evaluate)
default_timing_cols = ['ICP_Init_Time', 'ICP_Calibrate_Time', 'Predict_Time', 'Evaluate_Time', 'Case_Total_Time']
df_default_summary = df_default.groupby(['Device', 'Measure'])[default_timing_cols].agg(['mean', 'std']).round(6)

print(f"\n--- Default Case (W_H=0, W_C=0) Timing Summary ---")
print(df_default_summary.to_string())

# Summary for penalty update cases
penalty_timing_cols = ['Recalibration_Time', 'Predict_Time', 'Evaluate_Time', 'Case_Total_Time']
df_penalty_summary = df_penalty.groupby(['Device', 'Measure', 'Weight_Case'])[penalty_timing_cols].agg(
    ['mean', 'std']).round(6)

print(f"\n--- Penalty Update Cases Timing Summary ---")
print(df_penalty_summary.to_string())

# Combined summary for comparison
print(f"\n--- Combined Timing Comparison ---")
df_default_avg = df_default.groupby(['Device', 'Measure'])[default_timing_cols].mean()
df_penalty_avg = df_penalty.groupby(['Device', 'Measure', 'Weight_Case'])[penalty_timing_cols].mean()
print("\nDefault Case (W_H=0, W_C=0):")
print(df_default_avg.to_string())
print("\nPenalty Update Cases:")
print(df_penalty_avg.to_string())

# --- 4. Coverage and Size Results ---
plot_device = 'cuda' if 'cuda' in devices else 'cpu'

df_coverage = pd.DataFrame(index=alphas)
df_sizes = pd.DataFrame(index=alphas)

for case_key in all_cases:
    df_coverage[case_key] = coverage_results[plot_device][case_key].mean(axis=0)
    df_sizes[case_key] = size_results[plot_device][case_key].mean(axis=0)

df_coverage.index.name = 'Alpha'
df_sizes.index.name = 'Alpha'

# --- 5. Save All Results ---
print("\nSaving all results to CSV files...")

# Raw timing data (for flexible re-analysis)
df_default.to_csv(f"{out_dir}/raw_timing_default_{dataset}.csv", index=False)
df_penalty.to_csv(f"{out_dir}/raw_timing_penalty_{dataset}.csv", index=False)

# Timing summaries
df_default_summary.to_csv(f"{out_dir}/timing_default_case_{dataset}.csv")
df_penalty_summary.to_csv(f"{out_dir}/timing_penalty_cases_{dataset}.csv")

# Coverage and size results
df_coverage.to_csv(f"{out_dir}/empirical_validity_{dataset}.csv")
df_sizes.to_csv(f"{out_dir}/prediction_sizes_{dataset}.csv")


total_end_time = time.perf_counter()
print(f"\n{'='*60}")
print(f"All experiments finished successfully!")
print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
print(f"Results saved to: ./{out_dir}/")
print(f"{'='*60}")
print("\nSaved files:")
print(f"  - raw_timing_default_{dataset}.csv")
print(f"  - raw_timing_penalty_{dataset}.csv")
print(f"  - timing_default_case_{dataset}.csv")
print(f"  - timing_penalty_cases_{dataset}.csv")
print(f"  - empirical_validity_{dataset}.csv")
print(f"  - prediction_sizes_{dataset}.csv")
print(f"  - experiment_metadata.json")
# print(f"\nRun 'python plot_results.py' to generate plots.")


paper_data_dir = os.path.join(out_dir, 'paper_data')
if not os.path.exists(paper_data_dir):
    os.makedirs(paper_data_dir)


print('\nGenerating tables...')
# ==============================================================
#     Tex Table Prediction Region Percentages
# ===============================================================
df_sizes_restricted = df_sizes.loc[[0.01, 0.05, 0.1, 0.2]]
df_sizes_restricted['Min'] = df_sizes_restricted.min(axis=1)
df_sizes_restricted['Max'] = df_sizes_restricted.max(axis=1)
df_sizes_restricted = df_sizes_restricted[['Min', 'Max']]

df_sizes_restricted['Min_Pct'] = (((2 ** classes - df_sizes_restricted['Min']) / 2 ** classes) * 100)
df_sizes_restricted['Max_Pct'] = (((2 ** classes - df_sizes_restricted['Max']) / 2 ** classes) * 100)
df_sizes_restricted.drop(columns=['Min', 'Max'], inplace=True)
df_sizes_restricted.to_latex(os.path.join(paper_data_dir, 'predidiction_sizes_percentages.tex'),
                             index=True)

print('\nTex Table Prediction Region Percentages finished')

# ===============================================================================
#  Tex Table Penalty Timing per Device
# ================================================================================
df_penalty.drop(inplace=True, columns=['Iteration', 'Measure', 'Weight_Case'])
df_penalty.drop(df_penalty[(df_penalty['W_H'] == 1.0) & (df_penalty['W_C'] == 1.0)].index, inplace=True)
df_hamming = df_penalty[(df_penalty['W_H'] == 1.0) & (df_penalty['W_C'] == 0.0)].groupby('Device')[
    ['Recalibration_Time', 'Predict_Time', 'Evaluate_Time',
     'Case_Total_Time']].mean().copy()

df_cardinality = df_penalty[(df_penalty['W_H'] == 0.0) & (df_penalty['W_C'] == 1.0)].groupby('Device')[
    ['Recalibration_Time', 'Predict_Time', 'Evaluate_Time',
     'Case_Total_Time']].mean().copy()

df_hamming['Penalty'] = 'Hamming'
df_cardinality['Penalty'] = 'Cardinality'

df_combined = pd.concat([df_hamming.reset_index(), df_cardinality.reset_index()])
df_combined = df_combined[
    ['Penalty', 'Device', 'Recalibration_Time', 'Predict_Time', 'Evaluate_Time', 'Case_Total_Time']]

df_combined.to_latex(os.path.join(paper_data_dir, 'penalty_avg_timing.tex'),
                     index=False,
                     float_format='%.3f')

print('\nTex Table Penalty Timing per Device finished')

# ===============================================================================
#  Tex Table Default Timing per Device
# ================================================================================
df_default.drop(inplace=True, columns=['Iteration', 'Measure', 'Weight_Case', 'W_H', 'W_C'])
df_default_paper = df_default.groupby('Device')[
    ['ICP_Init_Time', 'ICP_Calibrate_Time', 'Predict_Time', 'Evaluate_Time', 'Case_Total_Time']].mean().round(
    3).reset_index()
df_default_paper.to_latex(os.path.join(paper_data_dir, 'default_avg_timing_per_device.tex'),
                          index=False,
                          float_format='%.3f')

print('\nTex Table Default Timing per Device finished')

# --- Plotting Configuration ---
colors_mah = ['orange', 'teal', 'crimson', 'gold']
colors_norm = ['gray', 'blue', 'green', 'purple']
markers = ['o', 's', '^', 'd']

print("\nGenerating plots...")

# =============================================================================
# Plot 1: Prediction Region Sizes
# =============================================================================
# Figure size
plt.figure(figsize=(7, 5.5))

for case_key in all_cases:
    measure_type, weight_type = case_key.split(" | ")
    line_style = '-' if measure_type == 'MAH' else '--'
    colors = colors_mah if measure_type == 'MAH' else colors_norm
    color_idx = list(all_weight_cases.keys()).index(weight_type)

    # Linewidth and markersize
    plt.plot(alphas, df_sizes[case_key], marker=markers[color_idx], markersize=7,
             linewidth=2.5, linestyle=line_style, color=colors[color_idx],
             label=case_key, alpha=0.8)

# Font sizes
plt.xlabel('Significance Level ', fontsize=16)
plt.ylabel('Average Prediction Region Size (N-Criterion)', fontsize=16)
plt.title(f'{dataset.capitalize()} Dataset', fontsize=18)

# Axis tick numbers
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.xlim([0.0, 0.2])
plt.grid(True, linestyle=':', alpha=0.7)

# Legend
plt.legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{paper_data_dir}/plot_prediction_sizes_{dataset}_paper.png", dpi=300)
plt.close()

print("  - Plot 1: Prediction Region Sizes finished")

# =============================================================================
# Plot 2: Empirical Validity
# =============================================================================
# Figure size (ideal for 0-to-1 calibration curves)
plt.figure(figsize=(7, 7))

for case_key in all_cases:
    measure_type, weight_type = case_key.split(" | ")
    line_style = '-' if measure_type == 'MAH' else '--'
    colors = colors_mah if measure_type == 'MAH' else colors_norm
    color_idx = list(all_weight_cases.keys()).index(weight_type)

    # Linewidth and markersize
    plt.plot(alphas, df_coverage[case_key], marker=markers[color_idx], markersize=7,
             linewidth=1.5, linestyle=line_style, color=colors[color_idx], label=case_key, alpha=0.8)

# Calibration line
plt.plot([0, 1], [1, 0], linestyle=':', color='red', linewidth=3, label='Perfect Calibration')

# Font size
plt.xlabel('Significance Level', fontsize=16)
plt.ylabel('Empirical Validity (Coverage)', fontsize=16)
plt.title(f'{dataset.capitalize()} Dataset', fontsize=18)

# Axis tick numbers
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid(True, linestyle=':', alpha=0.7)

# Legend
plt.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9)

plt.tight_layout()
plt.savefig(f"{paper_data_dir}/plot_empirical_validity_{dataset}.png", dpi=300)
plt.close()

print("  - Plot 2: Empirical Validity finished")