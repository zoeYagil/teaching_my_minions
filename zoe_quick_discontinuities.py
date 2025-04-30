'''
Play with:
    
    window_length = 5  # SG 
    polyorder = 2 #SG
    peak_threshold = 15 #analyze_discontinuities
    trough_threshold = 15# analyze_discontinuities
    magnitude_threshold = 2#analyze_discontinuities
    
In the second half you create a correlation matrix plot using a spearman coef with the integrated areas in a folder 
1101_to_3999. Error can happen if this doesn´t exist (!)

I am adding as well the alignment cost after using DTW on the normalized curve segments

'''





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import spearmanr
import seaborn as sns

import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'


#%% THINGS TO CHANGE
file_path = '/Users/danielsinausia/Documents/Experiments/CZ_00028_LLP/PCA_scores.txt'

data = pd.read_csv(file_path, delim_whitespace=True, header=None)

window_length = 60
polyorder = 2
peak_threshold = 0.1
trough_threshold = 0.1
magnitude_threshold = 0.15

folder = os.path.dirname(file_path)
folder_path = os.path.join(folder, 'PC1_discontinuities')
os.makedirs(folder_path, exist_ok=True)
filename = os.path.basename(folder) + "_PC1_discontinuities"

time = data[0].values[:] * 1.1
pc1 = data[1].values[:]
pc1_smoothed = savgol_filter(pc1, window_length=window_length, polyorder=polyorder)

def analyze_discontinuities(time, pc1_smoothed, peak_threshold, trough_threshold, magnitude_threshold):
    peaks, _ = find_peaks(pc1_smoothed, prominence=peak_threshold)
    troughs, _ = find_peaks(-pc1_smoothed, prominence=trough_threshold)
    discontinuities = np.sort(np.concatenate((peaks, troughs)))
    magnitudes = []

    for i in discontinuities:
        if i > 0 and i < len(pc1_smoothed) - 1:
            magnitude_before = abs(pc1_smoothed[i] - pc1_smoothed[i-1])
            magnitude_after = abs(pc1_smoothed[i] - pc1_smoothed[i+1])
            magnitudes.append(max(magnitude_before, magnitude_after))

    magnitudes = np.array(magnitudes)
    valid_indices = magnitudes >= magnitude_threshold
    discontinuities = discontinuities[valid_indices]
    magnitudes = magnitudes[valid_indices]

    return time[discontinuities], magnitudes, pc1_smoothed[discontinuities]

disc_times, disc_magnitudes, disc_values = analyze_discontinuities(
    time, pc1_smoothed, peak_threshold, trough_threshold, magnitude_threshold
)

plt.figure(figsize=(12, 8))
plt.plot(time, pc1_smoothed, label='PC1 Smoothed', linewidth=2, color='blue')
plt.scatter(disc_times, disc_values, label='Discontinuities', color='red', edgecolor='black', s=100)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('PC1', fontsize=14)
plt.title('PC1 with Identified Discontinuities', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

png_path = os.path.join(folder_path, f"{filename}.png")
svg_path = os.path.join(folder_path, f"{filename}.svg")
plt.savefig(png_path, dpi=300)
plt.savefig(svg_path, format='svg', transparent=True)
plt.show()

metadata = {
    'SG Filter Window Length': [window_length],
    'SG Filter Polyorder': [polyorder],
    'Peak Threshold': [peak_threshold],
    'Trough Threshold': [trough_threshold],
    'Magnitude Threshold': [magnitude_threshold]
}
metadata_df = pd.DataFrame(metadata)

discontinuity_info = pd.DataFrame({
    'Discontinuity Time (s)': disc_times,
    'Discontinuity Magnitude': disc_magnitudes,
    'Discontinuity Value': disc_values
})

csv_filename = os.path.join(folder_path, 'Discontinuities_Metadata.csv')
with open(csv_filename, 'w') as f:
    metadata_df.to_csv(f, index=False)
    f.write("\n")
    discontinuity_info.to_csv(f, index=False)

print(f"CSV report saved to: {csv_filename}")
