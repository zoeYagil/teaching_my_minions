import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d

plt.rcParams["font.family"] = "Arial"

# --- Configurable Parameters ---
spectrum_interval = 1.1
latent_dim = 8
autoencoder_epochs = 10
START_ANALYSIS_AT = 150.0
END_ANALYSIS_AT = 1000.0
SMOOTH_DATA = True
SMOOTHING_WINDOW = 5
SHOW_DISCONTINUITY_PLOTS = False
PLOT_SPECTRAL_DIFFS = False
DEBUG_PLOTS = True  # will only plot for first signal
MAX_LATENT_DIMS = latent_dim  # set to smaller value for testing (e.g. 2)

# --- Discontinuity Detection Parameters ---
# Updated parameters to match the new code
SG_WINDOW_LENGTH = 60  # Updated from 10 to 60
SG_POLYORDER = 2  # Kept the same as it was already 2
PEAK_THRESHOLD = 0.1  # Updated to match new code
TROUGH_THRESHOLD = 0.1  # Updated to match new code
MAGNITUDE_THRESHOLD = 0.15  # Updated to match new code

# --- File Paths ---
file_paths = [
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/CZ_00023 - 40 rpm.csv"
]

outpath = '/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/Discontinuity counting and analysis with ML/Serie 3 - with 9 datasets/EachIndividually/ZY_00023 - 40 rpm_discontinuities_summary.csv'

# --- Load Data ---
all_data, labels, time_indices, sample_ids = [], [], [], []
wavenumbers = None
min_wavenumber_count = None

for f in file_paths:
    df = pd.read_csv(f)
    if min_wavenumber_count is None or df.shape[0] < min_wavenumber_count:
        min_wavenumber_count = df.shape[0]

for i, fpath in enumerate(file_paths):
    df = pd.read_csv(fpath).iloc[:min_wavenumber_count, :]
    if wavenumbers is None:
        wavenumbers = df.iloc[:, 0].values
    spectra = df.drop(df.columns[0], axis=1).T.values
    spectra -= spectra.mean(axis=0)

    time_idx = np.arange(spectra.shape[0]) * spectrum_interval
    start_index = int(np.ceil(START_ANALYSIS_AT / spectrum_interval))
    end_index = int(np.floor(END_ANALYSIS_AT / spectrum_interval))

    spectra = spectra[start_index:end_index]
    time_idx = time_idx[start_index:end_index]

    if SMOOTH_DATA:
        spectra = uniform_filter1d(spectra, size=SMOOTHING_WINDOW, axis=0, mode='nearest')

    all_data.append(spectra)
    labels.extend([os.path.basename(fpath)] * spectra.shape[0])
    time_indices.extend(time_idx)
    sample_ids.extend([i] * spectra.shape[0])

X = np.vstack(all_data)
labels = np.array(labels)
time_indices = np.array(time_indices)
sample_ids = np.array(sample_ids)

# --- Normalize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Autoencoder ---
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
latent = Dense(latent_dim, activation='linear')(encoded)
decoded = Dense(32, activation='relu')(latent)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder = Model(inputs=input_layer, outputs=latent)
autoencoder.compile(optimizer=Adam(1e-3), loss='mse')

print("🚀 Training Autoencoder...")
autoencoder.fit(X_scaled, X_scaled, epochs=autoencoder_epochs, batch_size=32, verbose=1)
latent_features = encoder.predict(X_scaled)

# --- Discontinuity Detection ---
# Updated function to match the new code's approach
def analyze_discontinuities(signal, times, sg_window, sg_poly, peak_thresh, trough_thresh, magnitude_thresh, debug=False):
    smoothed = savgol_filter(signal, window_length=sg_window, polyorder=sg_poly)
    
    # Find peaks and troughs
    peaks, _ = find_peaks(smoothed, prominence=peak_thresh)
    troughs, _ = find_peaks(-smoothed, prominence=trough_thresh)
    
    # Combine and sort discontinuities
    discontinuities = np.sort(np.concatenate((peaks, troughs)))
    
    # Calculate magnitudes
    magnitudes = []
    for i in discontinuities:
        if i > 0 and i < len(smoothed) - 1:
            magnitude_before = abs(smoothed[i] - smoothed[i-1])
            magnitude_after = abs(smoothed[i] - smoothed[i+1])
            magnitudes.append(max(magnitude_before, magnitude_after))
    
    magnitudes = np.array(magnitudes)
    
    valid_indices = magnitudes >= magnitude_thresh
    discontinuities = discontinuities[valid_indices]
    magnitudes = magnitudes[valid_indices]
    
    disc_times = times[discontinuities] if times is not None else discontinuities
    disc_values = smoothed[discontinuities]
    
    print(f"→ Found {len(discontinuities)} discontinuities")

    if debug:
        plt.figure(figsize=(10, 3))
        plt.plot(smoothed, label='Smoothed')
        plt.scatter(discontinuities, smoothed[discontinuities], color='red', label='Discontinuities')
        plt.legend()
        plt.title("DEBUG: Discontinuity Detection")
        plt.tight_layout()
        plt.show()

    return discontinuities, magnitudes, disc_values, disc_times

# --- Main Loop ---
all_discontinuities = []
t0 = time.time()
DEBUG_SHOWN = False

for dim in range(min(latent_dim, MAX_LATENT_DIMS)):
    for sid in np.unique(sample_ids):
        idx = sample_ids == sid
        signal = latent_features[idx, dim]
        times = time_indices[idx]
        
        if len(signal) < SG_WINDOW_LENGTH:
            print(f"Signal too short for smoothing window (length={len(signal)}, window={SG_WINDOW_LENGTH}). Skipping.")
            continue

        jump_indices, jump_magnitudes, jump_values, jump_times = analyze_discontinuities(
            signal,
            times,
            SG_WINDOW_LENGTH,
            SG_POLYORDER,
            PEAK_THRESHOLD,
            TROUGH_THRESHOLD,
            MAGNITUDE_THRESHOLD,
            debug=(DEBUG_PLOTS and not DEBUG_SHOWN)
        )

        DEBUG_SHOWN = True  # Only show debug plot once

        if len(jump_indices) == 0:
            continue

        label = np.unique(labels[idx])[0]
        print(f"\n📍 Sample: {label}, Latent Dim Z{dim+1} – {len(jump_indices)} discontinuities")

        plt.figure(figsize=(8, 3))
        # Plot with actual times instead of indices
        plt.plot(times, signal, label=f'Z{dim+1} - {label}')
        plt.scatter(jump_times, jump_values, color='red', label='Discontinuities', edgecolor='black', s=50)
        plt.title(f'Discontinuities in Z{dim+1} – {label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Latent Value')
        plt.legend()
        plt.tight_layout()

        safe_label = label.replace(" ", "_").replace(".csv", "").replace("/", "_")
        plot_basename = f"discontinuities_Z{dim+1}_{safe_label}"
        plot_folder = os.path.dirname(outpath)
        os.makedirs(plot_folder, exist_ok=True)

        if SHOW_DISCONTINUITY_PLOTS:
            plt.show()
        else:
            plt.savefig(os.path.join(plot_folder, f"{plot_basename}.png"), dpi=300)
            plt.savefig(os.path.join(plot_folder, f"{plot_basename}.svg"), format='svg', transparent=True)
            plt.close()

        metadata = {
            'SG Filter Window Length': [SG_WINDOW_LENGTH],
            'SG Filter Polyorder': [SG_POLYORDER],
            'Peak Threshold': [PEAK_THRESHOLD],
            'Trough Threshold': [TROUGH_THRESHOLD],
            'Magnitude Threshold': [MAGNITUDE_THRESHOLD]
        }
        
        metadata_df = pd.DataFrame(metadata)
        discontinuity_info = pd.DataFrame({
            'Discontinuity Time (s)': jump_times,
            'Discontinuity Magnitude': jump_magnitudes,
            'Discontinuity Value': jump_values
        })
        
        metadata_path = os.path.join(plot_folder, f"{plot_basename}_metadata.csv")
        with open(metadata_path, 'w') as f:
            metadata_df.to_csv(f, index=False)
            f.write("\n")
            discontinuity_info.to_csv(f, index=False)
        
        for j, mag, val, t in zip(jump_indices, jump_magnitudes, jump_values, jump_times):
            all_discontinuities.append({
                'file': label,
                'sample_id': sid,
                'latent_dim': dim + 1,
                'index': int(j),
                'time_s': float(t),
                'magnitude': float(mag),
                'value': float(val)
            })

# --- Save CSV ---
dis_df = pd.DataFrame(all_discontinuities)
dis_df.to_csv(outpath, index=False)
print(f"\n✅ Saved discontinuity timestamps to: {outpath}")
print(f"⏱️ Total runtime: {time.time() - t0:.1f} seconds")
