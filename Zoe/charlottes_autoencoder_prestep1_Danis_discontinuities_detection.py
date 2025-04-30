import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter, find_peaks

# --- Configurable Parameters ---
spectrum_interval = 1.1
autoencoder_epochs = 10
START_ANALYSIS_AT = 150.0
PLOT_SPECTRAL_DIFFS = False
SMOOTH_DATA = True
SMOOTHING_WINDOW = 5
latent_dims_to_test = [2, 4, 8, 16, 32]

# --- Discontinuity Detection Parameters ---
SG_WINDOW_LENGTH = 60
SG_POLYORDER = 2
PEAK_THRESHOLD = 0.1
TROUGH_THRESHOLD = 0.1
MAGNITUDE_THRESHOLD = 0.15

# --- File paths ---
file_paths = [
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/CZ_00026 - 0 rpm.csv",
    
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/ZY_00009 - 5 rpm.csv",
    
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/ZY_00002 - 10 rpm.csv",
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/ZY_00003 - 10 rpm.csv",
    
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/ZY_00005 - 20 rpm.csv",
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/ZY_00007 - 20 rpm.csv",
    
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/ZY_00008 - 30 rpm.csv",
    
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/ZY_00004 - 40 rpm.csv",
    "/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/More Data/CZ_00023 - 40 rpm.csv"
]


outpath = '/Users/charlottevogt/Documents/Manuscripts/Emergent Behavior Discontinuities - Periodicity/ML Model Test April 2025/Discontinuity counting and analysis with ML/Testing How Many Latent Dimensions/output.csv'
plot_folder = os.path.dirname(outpath)

# --- Define the new discontinuity detection function ---
def analyze_discontinuities(signal, sg_window, sg_poly, peak_thresh, trough_thresh, magnitude_thresh):
    """
    Analyze signal for discontinuities using the approach from the reference code.
    
    Args:
        signal: The signal to analyze
        sg_window: Savitzky-Golay filter window length
        sg_poly: Savitzky-Golay filter polynomial order
        peak_thresh: Prominence threshold for peaks
        trough_thresh: Prominence threshold for troughs
        magnitude_thresh: Threshold for discontinuity magnitude
        
    Returns:
        discontinuities: Indices of detected discontinuities
        magnitudes: Magnitude values of discontinuities
    """
    # Check if signal is long enough for the window
    if len(signal) < sg_window:
        return np.array([]), np.array([])
    
    # Apply Savitzky-Golay filter
    smoothed = savgol_filter(signal, window_length=sg_window, polyorder=sg_poly)
    
    # Find peaks and troughs
    peaks, _ = find_peaks(smoothed, prominence=peak_thresh)
    troughs, _ = find_peaks(-smoothed, prominence=trough_thresh)
    
    # Combine and sort discontinuities
    discontinuities = np.sort(np.concatenate((peaks, troughs)))
    
    # If no discontinuities found, return empty arrays
    if len(discontinuities) == 0:
        return np.array([]), np.array([])
    
    # Calculate magnitudes
    magnitudes = []
    for i in discontinuities:
        if i > 0 and i < len(smoothed) - 1:
            magnitude_before = abs(smoothed[i] - smoothed[i-1])
            magnitude_after = abs(smoothed[i] - smoothed[i+1])
            magnitudes.append(max(magnitude_before, magnitude_after))
    
    magnitudes = np.array(magnitudes)
    
    # Filter by magnitude threshold
    valid_indices = magnitudes >= magnitude_thresh
    discontinuities = discontinuities[valid_indices]
    magnitudes = magnitudes[valid_indices]
    
    return discontinuities, magnitudes

# --- Load Data ---
all_data, labels, time_indices, sample_ids = [], [], [], []
wavenumbers = None

min_wavenumber_count = None
for f in file_paths:
    df = pd.read_csv(f)
    current_count = df.shape[0]
    if min_wavenumber_count is None or current_count < min_wavenumber_count:
        min_wavenumber_count = current_count

for i, fpath in enumerate(file_paths):
    df = pd.read_csv(fpath)
    df = df.iloc[:min_wavenumber_count, :]

    if wavenumbers is None:
        wavenumbers = df.iloc[:, 0].values
    else:
        wavenumbers = wavenumbers[:min_wavenumber_count]

    spectra = df.drop(df.columns[0], axis=1).T.values
    spectra = spectra - spectra.mean(axis=0)

    time_idx = np.arange(spectra.shape[0]) * spectrum_interval
    start_index = int(np.ceil(START_ANALYSIS_AT / spectrum_interval))
    spectra = spectra[start_index:]
    time_idx = time_idx[start_index:]

    if SMOOTH_DATA and SMOOTHING_WINDOW > 1:
        spectra = uniform_filter1d(spectra, size=SMOOTHING_WINDOW, axis=0, mode='nearest')

    all_data.append(spectra)
    labels.extend([os.path.basename(fpath)] * spectra.shape[0])
    time_indices.extend(time_idx)
    sample_ids.extend([i] * spectra.shape[0])

X = np.vstack(all_data)
labels = np.array(labels)
sample_ids = np.array(sample_ids)
X_scaled = StandardScaler().fit_transform(X)

# --- Create a dataframe to store results ---
results_data = []

# --- Autoencoder sweep ---
reconstruction_losses = []
discontinuity_counts = []

for latent_dim in latent_dims_to_test:
    print(f"\n🔍 Training Autoencoder with latent_dim = {latent_dim}...")
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    latent = Dense(latent_dim, activation='linear')(encoded)
    decoded = Dense(32, activation='relu')(latent)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(1e-3), loss='mse')

    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=autoencoder_epochs,
        batch_size=32,
        verbose=0
    )

    final_loss = history.history['loss'][-1]
    reconstruction_losses.append(final_loss)

    encoder = Model(inputs=input_layer, outputs=latent)
    latent_features = encoder.predict(X_scaled)

    total_discontinuities = 0
    sample_discontinuities = {}
    
    # Initialize discontinuity counts for each sample
    for sid in np.unique(sample_ids):
        sample_discontinuities[sid] = 0
    
    # Detect discontinuities using the new method
    for dim in range(latent_features.shape[1]):
        for sid in np.unique(sample_ids):
            idx = sample_ids == sid
            signal = latent_features[idx, dim]
            
            # Skip if signal is too short
            if len(signal) < SG_WINDOW_LENGTH:
                continue
                
            # Use the new discontinuity detection function
            jump_indices, jump_magnitudes = analyze_discontinuities(
                signal,
                SG_WINDOW_LENGTH,
                SG_POLYORDER,
                PEAK_THRESHOLD,
                TROUGH_THRESHOLD,
                MAGNITUDE_THRESHOLD
            )
            
            disc_count = len(jump_indices)
            total_discontinuities += disc_count
            sample_discontinuities[sid] += disc_count
            
            # If we want to save specific discontinuity information, we could add it here
            if disc_count > 0:
                sample_name = np.unique(labels[idx])[0]
                # Convert boolean mask to indices first
                idx_array = np.where(idx)[0]
                # Then get the time indices for these points
                time_indices_for_sample = time_indices[idx_array]
                # Finally index into these with jump_indices
                times = time_indices_for_sample[jump_indices]
                
                for j, mag, t in zip(jump_indices, jump_magnitudes, times):
                    results_data.append({
                        'latent_dim': latent_dim,
                        'latent_feature': dim + 1,
                        'sample': sample_name,
                        'sample_id': sid,
                        'time_index': int(j),
                        'time_s': float(t),
                        'magnitude': float(mag)
                    })

    discontinuity_counts.append(total_discontinuities)
    print(f"✅ latent_dim={latent_dim} → Discontinuities Detected: {total_discontinuities}, Loss: {final_loss:.5f}")
    
    # Print breakdown by sample
    for sid in np.unique(sample_ids):
        idx = sample_ids == sid
        sample_name = np.unique(labels[idx])[0]
        print(f"  - {sample_name}: {sample_discontinuities[sid]} discontinuities")

# Save detailed results
if results_data:
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(outpath, index=False)
    print(f"✅ Saved detailed discontinuity data to: {outpath}")

# --- Plot: Loss vs Latent Dim ---
plt.figure(figsize=(8, 5))
plt.plot(latent_dims_to_test, reconstruction_losses, marker='o')
plt.xlabel("Latent Dimension", fontsize=12)
plt.ylabel("Reconstruction Loss (MSE)", fontsize=12)
plt.title("Reconstruction Loss vs. Latent Dimension", fontsize=14)
plt.grid(True)
plt.xticks(latent_dims_to_test)
plt.rcParams["font.family"] = "Arial"
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "reconstruction_loss_vs_latent_dim.png"), dpi=300)
plt.savefig(os.path.join(plot_folder, "reconstruction_loss_vs_latent_dim.svg"), format="svg", dpi=300)
plt.close()

# --- Plot: Discontinuities vs Latent Dim ---
plt.figure(figsize=(8, 5))
plt.plot(latent_dims_to_test, discontinuity_counts, marker='o', color='darkred')
plt.xlabel("Latent Dimension", fontsize=12)
plt.ylabel("Detected Discontinuities", fontsize=12)
plt.title("Discontinuities vs. Latent Dimension", fontsize=14)
plt.grid(True)
plt.xticks(latent_dims_to_test)
plt.rcParams["font.family"] = "Arial"
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "discontinuities_vs_latent_dim.png"), dpi=300)
plt.savefig(os.path.join(plot_folder, "discontinuities_vs_latent_dim.svg"), format="svg", dpi=300)
plt.close()

# Create and save metadata file
metadata = {
    'SG Filter Window Length': [SG_WINDOW_LENGTH],
    'SG Filter Polyorder': [SG_POLYORDER],
    'Peak Threshold': [PEAK_THRESHOLD],
    'Trough Threshold': [TROUGH_THRESHOLD],
    'Magnitude Threshold': [MAGNITUDE_THRESHOLD],
    'Latent Dimensions Tested': [str(latent_dims_to_test)],
    'Total Discontinuities Detected': [sum(discontinuity_counts)]
}

metadata_df = pd.DataFrame(metadata)
metadata_path = os.path.join(plot_folder, "discontinuity_detection_parameters.csv")
metadata_df.to_csv(metadata_path, index=False)
print(f"✅ Saved detection parameters to: {metadata_path}")
