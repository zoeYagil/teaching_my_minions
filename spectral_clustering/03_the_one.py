'''

Warning: this doesn´t just normalize columns 1:16. This normalizes first only column 1, then columns 1:2, then columns 1:3, ... so resulting csv file has all these combinations

'''


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'




row_range = (80, 180)








def dtw_distance_with_path_and_cost(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.zeros((n + 1, m + 1)) + np.inf
    dtw[0, 0] = 0
    traceback = np.zeros((n, m), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            options = [dtw[i - 1, j],  # insertion
                       dtw[i, j - 1],  # deletion
                       dtw[i - 1, j - 1]]  # match
            tb_index = np.argmin(options)
            dtw[i, j] = cost + options[tb_index]
            traceback[i - 1, j - 1] = tb_index

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback[i, j]
        if tb_type == 0:  # insertion
            i -= 1
        elif tb_type == 1:  # deletion
            j -= 1
        else:  # match
            i -= 1
            j -= 1
        path.append((i, j))
    return dtw[n, m], path[::-1], dtw[1:, 1:]

def plot_dtw_matrices_and_path(dist_mat, cost_mat, path, x, y, output_folder, file_suffix):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplot(121)
    plt.title("Distance Matrix")
    plt.imshow(dist_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    plt.subplot(122)
    plt.title("Cost Matrix")
    plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    x_path, y_path = zip(*path)
    plt.plot(y_path, x_path)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"DTW_Matrices_{file_suffix}.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    for x_i, y_j in path:
        plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
    plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
    plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, f"DTW_Alignment_{file_suffix}.png"), dpi=300)
    plt.savefig(os.path.join(output_folder, f"DTW_Alignment_{file_suffix}.svg"), format="svg")
    plt.close()

def normalize_file_absolute(data):
    """
    Normalize all columns in the file together using absolute values for scaling.
    """
    max_abs_val = np.max(np.abs(data.values))
    return data / max_abs_val if max_abs_val != 0 else data

def smooth_sequence(seq, window_length=50, polyorder=2):
    return savgol_filter(seq, window_length=window_length, polyorder=polyorder)

def save_references_to_csv_and_plot(references, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    max_length = max(len(curve) for curve in references.values())
    padded_references = {name: np.pad(curve, (0, max_length - len(curve)), constant_values=np.nan) for name, curve in references.items()}
    references_df = pd.DataFrame(padded_references)
    csv_path = os.path.join(output_dir, "reference_curves.csv")
    references_df.to_csv(csv_path, index=False)
    print(f"Reference curves saved to: {csv_path}")

    plt.figure(figsize=(12, 8))
    for name, curve in references.items():
        plt.plot(curve, label=name)

    plt.title("Reference Curves")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend(loc="best", fontsize='small')
    plt.grid()
    plot_path = os.path.join(output_dir, "reference_curves_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Reference curves plot saved to: {plot_path}")
def normalize_to_magnitude_one(curve):
    max_val = np.max(np.abs(curve))
    return curve / max_val if max_val != 0 else curve


n_points = 100
n_flat = 9
step_size = 2

# Reference curves without flat ends
reference_1 = np.zeros(n_points)
exp_decay = np.exp(-np.arange(n_points - n_flat) / 10)
reference_2 = np.concatenate((np.zeros(n_flat), exp_decay - exp_decay[0]))
exp_increase = 1 - np.exp(-np.arange(n_points - n_flat) / 10)
reference_3 = np.concatenate((np.zeros(n_flat), exp_increase - exp_increase[0]))
reference_4 = np.concatenate((np.zeros(n_flat), np.full(n_points - n_flat, step_size)))
reference_5 = np.concatenate((np.zeros(n_flat), np.full(n_points - n_flat, -step_size)))
decreasing_linear = np.concatenate((np.zeros(n_flat), np.linspace(step_size, -step_size, n_points - n_flat)))
increasing_linear = np.concatenate((np.zeros(n_flat), np.linspace(-step_size, step_size, n_points - n_flat)))
sinusoidal_curve = np.concatenate((np.zeros(n_flat), np.sin(np.linspace(0, 2 * np.pi, n_points - n_flat))))
inverse_sinusoidal_curve = np.concatenate((np.zeros(n_flat), -np.sin(np.linspace(0, 2 * np.pi, n_points - n_flat))))

#the weird curve
def smooth_transition(x, midpoint, width):
    return 1 / (1 + np.exp(-(x - midpoint) / width))
increase_duration = 10
decay_rate = 0.1
baseline_shift = 2.0
transition_width = 3 
flat_section = np.zeros(n_flat)
x_increase = np.linspace(-6, 6, increase_duration)
smooth_increase = 1 / (1 + np.exp(-x_increase))
smooth_increase = (smooth_increase - smooth_increase[0]) / (smooth_increase[-1] - smooth_increase[0])

x_combined = np.linspace(0, n_points - n_flat, n_points - n_flat)
smooth_transition_curve = (
    smooth_increase[-1] * (1 - smooth_transition(x_combined, increase_duration, 2))
    + (np.exp(-decay_rate * x_combined) - baseline_shift) * smooth_transition(x_combined, increase_duration, 2)
)
smooth_transition_curve_final = np.concatenate((flat_section, smooth_increase, smooth_transition_curve))
# Compute the inverted version of the smooth transition curve
smooth_transition_curve_flipped = -smooth_transition_curve_final



# Additional reference curves (no flat lines)
decreasing_linear_no_flat = np.linspace(step_size, -step_size, n_points)
increasing_linear_no_flat = np.linspace(-step_size, step_size, n_points)





references = {
    "Flat horizontal line": normalize_to_magnitude_one(reference_1),
    "Exponential decay": normalize_to_magnitude_one(reference_2),
    "Exponential increase": normalize_to_magnitude_one(reference_3),
    "Step up": normalize_to_magnitude_one(reference_4),
    "Step down": normalize_to_magnitude_one(reference_5),
    "Decreasing linear curve with flat start": normalize_to_magnitude_one(decreasing_linear),
    "Increasing linear curve with flat start": normalize_to_magnitude_one(increasing_linear),
    "Decreasing linear curve": normalize_to_magnitude_one(decreasing_linear_no_flat),
    "Increasing linear curve": normalize_to_magnitude_one(increasing_linear_no_flat),
    "Sinusoidal curve": normalize_to_magnitude_one(sinusoidal_curve),
    "Inverse sinusoidal curve": normalize_to_magnitude_one(inverse_sinusoidal_curve),
    "Smooth transition (width=2)": normalize_to_magnitude_one(smooth_transition_curve_final),
    "Inverse smooth transition (width=2)": normalize_to_magnitude_one(smooth_transition_curve_flipped)
}




include_folders = {
    "DS_00132", "DS_00133", "DS_00134", "DS_00127", "DS_00163", "DS_00131",
    "DS_00138", "DS_00135", "DS_00139", "DS_00136", "DS_00140", "DS_00137",
    "DS_00141", "DS_00144", "DS_00142", "DS_00145", "DS_00143", "DS_00146",
    "DS_00181", "DS_00180", "DS_00148", "DS_00152", "DS_00149", "DS_00153",
    "CZ_00016", "CZ_00001", "CZ_00003", "CZ_00005", "CZ_00002", "CZ_00012", "CZ_00013", 
    "CZ_00007", "CZ_00008", "CZ_00011", "CZ_00015", "CZ_00014", "CZ_00010", "CZ_00009"

}
ignore_folders = {
    "raw data", "CO peak", "Stark shift", "2500_to_3999", "1001_to_3999",
    "1635_peak", "2000_to_3999", "900_to_3999", "650_to_4000",
    "Diffusion coefficient plots", "DS_00145_01", "First derivative",
    "non-mean-centered", "test", "650 to 4000"
}
base_dir = "/Users/danielsinausia/Documents/Experiments"
output_base_dir = os.path.join(base_dir, "DTW_Analysis_SG_Smoothed_new_curves_normalizing_per_txt")
os.makedirs(output_base_dir, exist_ok=True)

summary_results = []

for folder_name in include_folders:
    folder_path = os.path.join(base_dir, folder_name)
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d.lower() not in ignore_folders]
        for file_name in files:
            if file_name == "PCA_scores.txt":
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")
                try:
                    file_output_dir = os.path.join(output_base_dir, folder_name)
                    os.makedirs(file_output_dir, exist_ok=True)
                    subfolder_name = "first_pulse_short"
                    subfolder_path = os.path.join(file_output_dir, subfolder_name)
                    os.makedirs(subfolder_path, exist_ok=True)
                    data = pd.read_csv(file_path, delimiter="\t", header=None).iloc[row_range[0]:row_range[1], 1:16]
                    data_smoothed = data.apply(smooth_sequence, axis=0)
                    data_normalized = normalize_file_absolute(data_smoothed)
                    results = []
                    for col_idx in range(data_normalized.shape[1]):
                        variable = data_normalized.iloc[:, col_idx].values
                
                        for name, ref in references.items():
                            ref_normalized = normalize_to_magnitude_one(ref)
                            dist_mat = np.abs(np.subtract.outer(variable, ref_normalized))
                            dtw_distance_value, path, cost_matrix = dtw_distance_with_path_and_cost(variable, ref_normalized)
                
                            results.append({
                                "File": file_path,
                                "PC": col_idx + 1,
                                "Reference": name,
                                "DTW Distance": dtw_distance_value
                            })
                            plot_dtw_matrices_and_path(
                                dist_mat, cost_matrix, path,
                                variable, ref_normalized,
                                file_output_dir,
                                f"PC{col_idx + 1}_{name.replace(' ', '_')}"
                            )
                        best_match = min(results, key=lambda x: x["DTW Distance"])
                        best_match["Best Match"] = "Yes"
                        for result in results:
                            result["Best Match"] = "Yes" if result["Reference"] == best_match["Reference"] else "No"
                            summary_results.append(result)
                
                    results_df = pd.DataFrame(results)
                    results_csv_path = os.path.join(file_output_dir, "DTW_Results_first_pulse_short.csv")
                    results_df.to_csv(results_csv_path, index=False)
                    print(f"Results saved to: {results_csv_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
summary_df = pd.DataFrame(summary_results)
summary_csv_path = os.path.join(output_base_dir, "DTW_Summary_First pulse.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary results saved to: {summary_csv_path}")


references_output_dir = os.path.join(output_base_dir, "Reference_Curves")
save_references_to_csv_and_plot(references, references_output_dir)
