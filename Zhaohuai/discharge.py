import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import warnings

import matplotlib as mpl

# Set font settings for better compatibility with SVG text
mpl.rcParams['svg.fonttype'] = 'none'
mpl.use('SVG')


# Suppress RuntimeWarning for better output readability
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define the input and output paths
input_file = "/Users/danielsinausia/Documents/Experiments/DS_00145/PC2-15/PCA_scores.txt"
output_dir = "/Users/danielsinausia/Downloads/PCA_exp_fit_results"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Define exponential function
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Define multi-segment exponential function
def multi_exp_func(x, *params):
    y = np.zeros_like(x, dtype=float)
    segment_length = 91
    
    # Each segment requires 3 parameters (a, b, c)
    params_per_segment = 3
    num_segments = len(params) // params_per_segment
    
    for i in range(num_segments):
        # Get the parameters for this segment
        segment_params = params[i*params_per_segment:(i+1)*params_per_segment]
        a, b, c = segment_params
        
        # Define the segment range
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(x))
        
        # Skip if we're past the end of the data
        if start_idx >= len(x):
            break
        
        # Calculate segment's x values relative to segment start
        segment_x = x[start_idx:end_idx] - x[start_idx]  # Zero-based for each segment
        
        # Apply exponential function to this segment
        y[start_idx:end_idx] = a * np.exp(b * segment_x) + c
    
    return y

# Function to save plots in both PNG and SVG formats
def save_plot(plt, base_path):
    # Save as PNG
    png_path = f"{base_path}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as SVG
    svg_path = f"{base_path}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    return png_path, svg_path

# Load the data
try:
    # Try to load with pandas first
    data = pd.read_csv(input_file, sep=None, engine='python')
    x = np.arange(len(data.iloc[:, 1]))
    y = data.iloc[:, 1].values
except:
    # If pandas fails, try numpy loadtxt
    try:
        data = np.loadtxt(input_file)
        x = np.arange(len(data[:, 1]))
        y = data[:, 1]
    except:
        # Last resort: read as text and parse
        with open(input_file, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            try:
                values = line.strip().split()
                if len(values) > 1:
                    data.append(float(values[1]))
            except:
                continue
        x = np.arange(len(data))
        y = np.array(data)

print(f"Successfully loaded data with {len(y)} points")

# Calculate the number of segments
segment_length = 91
num_segments = (len(x) + segment_length - 1) // segment_length  # Ceiling division
print(f"Data will be fitted with {num_segments} exponential segments (every 91 points)")

# Dictionary to store all fit results for comparison
fit_results = {}

# Fit the single exponential curve (full data)
try:
    # Initial parameter guess
    p0 = [1.0, -0.1, 0.0]
    
    # Perform the curve fitting
    popt, pcov = curve_fit(exp_func, x, y, p0=p0)
    
    # Get the fitted parameters
    a, b, c = popt
    
    # Calculate the fitted curve
    y_fit_exp = exp_func(x, a, b, c)
    
    # Calculate R-squared for single exponential fit
    residuals_exp = y - y_fit_exp
    ss_res_exp = np.sum(residuals_exp**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared_exp = 1 - (ss_res_exp / ss_tot)
    
    print(f"Single exponential fit successful with parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print(f"Single exponential R-squared: {r_squared_exp:.4f}")
    
    # Store results
    fit_results['single_exponential'] = {
        'r_squared': r_squared_exp,
        'y_fit': y_fit_exp,
        'residuals': residuals_exp,
        'params': {'a': a, 'b': b, 'c': c},
        'equation': f"{a:.4f} * exp({b:.4f} * x) + {c:.4f}"
    }
    
except Exception as e:
    print(f"Error during single exponential curve fitting: {str(e)}")
    fit_results['single_exponential'] = None

# Fit individual exponential curves to each segment
segment_fits = []
segment_params = []
segment_r_squared = []

for i in range(num_segments):
    start_idx = i * segment_length
    end_idx = min((i + 1) * segment_length, len(x))
    
    # Skip if we're past the end of the data
    if start_idx >= len(x):
        break
    
    # Extract segment data
    segment_x = x[start_idx:end_idx]
    segment_y = y[start_idx:end_idx]
    
    # Convert to segment-relative x values (starting from 0)
    segment_x_rel = segment_x - segment_x[0]
    
    try:
        # Initial parameter guess
        p0 = [segment_y[0], -0.1, 0.0]
        
        # Perform the curve fitting for this segment
        popt, pcov = curve_fit(exp_func, segment_x_rel, segment_y, p0=p0)
        
        # Get the fitted parameters
        a, b, c = popt
        
        # Calculate the fitted curve
        segment_y_fit = exp_func(segment_x_rel, a, b, c)
        
        # Calculate R-squared for this segment
        segment_residuals = segment_y - segment_y_fit
        segment_ss_res = np.sum(segment_residuals**2)
        segment_ss_tot = np.sum((segment_y - np.mean(segment_y))**2)
        
        # Handle case where all y values in segment are identical
        if segment_ss_tot == 0:
            segment_r2 = 1.0  # Perfect fit
        else:
            segment_r2 = 1 - (segment_ss_res / segment_ss_tot)
        
        print(f"Segment {i+1}: Exponential fit successful with R-squared: {segment_r2:.4f}")
        
        # Store segment results
        segment_fits.append((segment_x, segment_y_fit))
        segment_params.append((a, b, c))
        segment_r_squared.append(segment_r2)
        
    except Exception as e:
        print(f"Error fitting segment {i+1}: {str(e)}")
        # Use a constant value as fallback
        a, b, c = np.mean(segment_y), 0, 0
        segment_y_fit = np.full_like(segment_x_rel, np.mean(segment_y))
        segment_fits.append((segment_x, segment_y_fit))
        segment_params.append((a, b, c))
        segment_r_squared.append(0)

# Fit the multi-segment exponential model
try:
    # Initial parameter guess - use the individual segment parameters
    p0 = []
    for a, b, c in segment_params:
        p0.extend([a, b, c])
    
    # Perform the curve fitting
    popt, pcov = curve_fit(multi_exp_func, x, y, p0=p0, maxfev=10000)
    
    # Calculate the fitted curve
    y_fit_multi = multi_exp_func(x, *popt)
    
    # Calculate R-squared for multi-segment exponential fit
    residuals_multi = y - y_fit_multi
    ss_res_multi = np.sum(residuals_multi**2)
    r_squared_multi = 1 - (ss_res_multi / ss_tot)
    
    print(f"Multi-segment exponential fit successful with {num_segments} segments")
    print(f"Multi-segment exponential R-squared: {r_squared_multi:.4f}")
    
    # Store results
    multi_segment_params = []
    for i in range(num_segments):
        if i*3+2 < len(popt):
            a, b, c = popt[i*3], popt[i*3+1], popt[i*3+2]
            multi_segment_params.append({'a': a, 'b': b, 'c': c})
    
    fit_results['multi_segment_exponential'] = {
        'r_squared': r_squared_multi,
        'y_fit': y_fit_multi,
        'residuals': residuals_multi,
        'params': multi_segment_params,
        'segment_length': segment_length
    }
    
except Exception as e:
    print(f"Error during multi-segment exponential curve fitting: {str(e)}")
    print("Using individual segment fits as fallback")
    
    # Create combined fit from individual segments
    y_fit_multi = np.zeros_like(y)
    for i, (segment_x, segment_y_fit) in enumerate(segment_fits):
        y_fit_multi[segment_x[0]:segment_x[-1]+1] = segment_y_fit
    
    # Calculate R-squared for combined individual fits
    residuals_multi = y - y_fit_multi
    ss_res_multi = np.sum(residuals_multi**2)
    r_squared_multi = 1 - (ss_res_multi / ss_tot)
    
    print(f"Combined individual segment fits R-squared: {r_squared_multi:.4f}")
    
    fit_results['multi_segment_exponential'] = {
        'r_squared': r_squared_multi,
        'y_fit': y_fit_multi,
        'residuals': residuals_multi,
        'params': segment_params,
        'segment_r_squared': segment_r_squared,
        'segment_length': segment_length
    }

# Determine the better model based on R-squared
if fit_results['single_exponential'] and fit_results['multi_segment_exponential']:
    better_model = 'single_exponential' if fit_results['single_exponential']['r_squared'] > fit_results['multi_segment_exponential']['r_squared'] else 'multi_segment_exponential'
    r_squared_diff = abs(fit_results['single_exponential']['r_squared'] - fit_results['multi_segment_exponential']['r_squared'])
    print(f"Better model based on R-squared: {better_model} (difference of {r_squared_diff:.4f})")
else:
    better_model = 'single_exponential' if fit_results['single_exponential'] else 'multi_segment_exponential'
    print(f"Only {better_model} fit was successful")

# Create comparison plot
plt.figure(figsize=(15, 10))

# Data points
plt.scatter(x, y, label='Data points', color='blue', alpha=0.5)

# Single exponential fit
if fit_results['single_exponential']:
    plt.plot(x, fit_results['single_exponential']['y_fit'], 'r-', 
             label=f"Single Exponential: {fit_results['single_exponential']['equation']}\nR² = {fit_results['single_exponential']['r_squared']:.4f}", 
             linewidth=2)

# Multi-segment exponential fit
if fit_results['multi_segment_exponential']:
    plt.plot(x, fit_results['multi_segment_exponential']['y_fit'], 'g-', 
             label=f"Multi-segment Exponential (91-point segments)\nR² = {fit_results['multi_segment_exponential']['r_squared']:.4f}", 
             linewidth=2)

plt.legend(fontsize=12, loc='best')
plt.title('Model Comparison: Single vs Multi-segment Exponential Fit', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('PCA Score (Column 2)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add R-squared comparison text on the plot
if fit_results['single_exponential'] and fit_results['multi_segment_exponential']:
    plt.text(0.02, 0.98, 
             f"Model comparison:\n"
             f"Single Exponential R² = {fit_results['single_exponential']['r_squared']:.4f}\n"
             f"Multi-segment Exp R² = {fit_results['multi_segment_exponential']['r_squared']:.4f}\n"
             f"Better model: {better_model} (+{r_squared_diff:.4f})", 
             transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save comparison plot in both formats
comparison_plot_base = os.path.join(output_dir, 'model_comparison_plot')
png_path, svg_path = save_plot(plt, comparison_plot_base)
print(f"Saved comparison plot to: {png_path} and {svg_path}")

# Create individual plots
# Single exponential fit
if fit_results['single_exponential']:
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, label='Data points', color='blue', alpha=0.7)
    plt.plot(x, fit_results['single_exponential']['y_fit'], 'r-', 
             label=f"Fit: {fit_results['single_exponential']['equation']}", linewidth=2)
    plt.legend(fontsize=12)
    plt.title('Single Exponential Fit of PCA Scores (Column 2)', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('PCA Score (Column 2)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'R² = {fit_results["single_exponential"]["r_squared"]:.4f}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    single_exp_base = os.path.join(output_dir, f'single_exponential_R2_{fit_results["single_exponential"]["r_squared"]:.4f}')
    png_path, svg_path = save_plot(plt, single_exp_base)
    print(f"Saved single exponential plot to: {png_path} and {svg_path}")

# Multi-segment exponential fit
if fit_results['multi_segment_exponential']:
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, label='Data points', color='blue', alpha=0.7)
    plt.plot(x, fit_results['multi_segment_exponential']['y_fit'], 'g-', 
             label=f"Multi-segment Exponential (91-point segments)", linewidth=2)
    
    # Add vertical lines to show segment boundaries
    for i in range(1, num_segments):
        boundary = i * segment_length
        if boundary < len(x):
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    plt.legend(fontsize=12)
    plt.title('Multi-segment Exponential Fit of PCA Scores (Column 2)', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('PCA Score (Column 2)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'R² = {fit_results["multi_segment_exponential"]["r_squared"]:.4f}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    multi_exp_base = os.path.join(output_dir, f'multi_segment_exponential_R2_{fit_results["multi_segment_exponential"]["r_squared"]:.4f}')
    png_path, svg_path = save_plot(plt, multi_exp_base)
    print(f"Saved multi-segment exponential plot to: {png_path} and {svg_path}")

# Create individual segment plots if multi-segment model is used
if fit_results['multi_segment_exponential'] and 'segment_r_squared' in fit_results['multi_segment_exponential']:
    # Create a directory for segment plots
    segments_dir = os.path.join(output_dir, 'segment_plots')
    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    
    # Plot each segment individually
    for i, (segment_x, segment_y_fit) in enumerate(segment_fits):
        plt.figure(figsize=(10, 6))
        
        # Plot original data points for this segment
        segment_y_orig = y[segment_x[0]:segment_x[-1]+1]
        plt.scatter(segment_x, segment_y_orig, color='blue', alpha=0.7, label='Data points')
        
        # Plot the fitted curve
        plt.plot(segment_x, segment_y_fit, 'g-', linewidth=2, 
                 label=f"Exponential fit (Segment {i+1})")
        
        # Add segment R-squared value
        r2 = fit_results['multi_segment_exponential']['segment_r_squared'][i]
        plt.text(0.05, 0.95, f'Segment {i+1} R² = {r2:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Get the parameters for this segment
        a, b, c = segment_params[i]
        plt.title(f'Segment {i+1}: Exponential Fit y = {a:.4f} * exp({b:.4f} * x) + {c:.4f}', fontsize=12)
        
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('PCA Score (Column 2)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save the segment plot
        segment_base = os.path.join(segments_dir, f'segment_{i+1}_R2_{r2:.4f}')
        png_path, svg_path = save_plot(plt, segment_base)
        print(f"Saved segment {i+1} plot to: {png_path} and {svg_path}")

# Save comprehensive results to CSV
results_df = pd.DataFrame({
    'Index': x,
    'Original_Data': y,
})

if fit_results['single_exponential']:
    results_df['Single_Exponential_Fitted_Values'] = fit_results['single_exponential']['y_fit']
    results_df['Single_Exponential_Residuals'] = fit_results['single_exponential']['residuals']

if fit_results['multi_segment_exponential']:
    results_df['Multi_Segment_Exponential_Fitted_Values'] = fit_results['multi_segment_exponential']['y_fit']
    results_df['Multi_Segment_Exponential_Residuals'] = fit_results['multi_segment_exponential']['residuals']

csv_path = os.path.join(output_dir, 'comparison_results.csv')
results_df.to_csv(csv_path, index=False)
print(f"Saved comprehensive results to: {csv_path}")

# Save parameters and comparison to text file
params_path = os.path.join(output_dir, 'fit_comparison.txt')
with open(params_path, 'w') as f:
    f.write(f"MODEL COMPARISON: SINGLE VS MULTI-SEGMENT EXPONENTIAL\n")
    f.write(f"=================================================\n\n")
    
    # Single exponential model details
    if fit_results['single_exponential']:
        f.write(f"SINGLE EXPONENTIAL MODEL: y = a * exp(b * x) + c\n")
        f.write(f"Parameters:\n")
        f.write(f"a = {fit_results['single_exponential']['params']['a']:.6f}\n")
        f.write(f"b = {fit_results['single_exponential']['params']['b']:.6f}\n")
        f.write(f"c = {fit_results['single_exponential']['params']['c']:.6f}\n")
        f.write(f"R-squared (R²): {fit_results['single_exponential']['r_squared']:.6f}\n")
        f.write(f"Equation: y = {fit_results['single_exponential']['equation']}\n\n")
    else:
        f.write(f"SINGLE EXPONENTIAL MODEL: Fitting failed\n\n")
    
    # Multi-segment exponential model details
    if fit_results['multi_segment_exponential']:
        f.write(f"MULTI-SEGMENT EXPONENTIAL MODEL: (segments of {segment_length} points)\n")
        f.write(f"Parameters for each segment (a, b, c for y = a * exp(b * x) + c):\n")
        
        # If we have segment-specific R² values, include them
        if 'segment_r_squared' in fit_results['multi_segment_exponential']:
            for i, params in enumerate(fit_results['multi_segment_exponential']['params']):
                if isinstance(params, tuple):
                    a, b, c = params
                    r2 = fit_results['multi_segment_exponential']['segment_r_squared'][i]
                    f.write(f"Segment {i+1}: a = {a:.6f}, b = {b:.6f}, c = {c:.6f}, R² = {r2:.6f}\n")
                else:
                    a, b, c = params['a'], params['b'], params['c']
                    f.write(f"Segment {i+1}: a = {a:.6f}, b = {b:.6f}, c = {c:.6f}\n")
        else:
            for i, params in enumerate(fit_results['multi_segment_exponential']['params']):
                if isinstance(params, dict):
                    a, b, c = params['a'], params['b'], params['c']
                    f.write(f"Segment {i+1}: a = {a:.6f}, b = {b:.6f}, c = {c:.6f}\n")
        
        f.write(f"\nOverall R-squared (R²): {fit_results['multi_segment_exponential']['r_squared']:.6f}\n\n")
    else:
        f.write(f"MULTI-SEGMENT EXPONENTIAL MODEL: Fitting failed\n\n")
    
    # Model comparison
    if fit_results['single_exponential'] and fit_results['multi_segment_exponential']:
        r_squared_diff = abs(fit_results['single_exponential']['r_squared'] - fit_results['multi_segment_exponential']['r_squared'])
        f.write(f"MODEL COMPARISON:\n")
        f.write(f"Single Exponential R² = {fit_results['single_exponential']['r_squared']:.6f}\n")
        f.write(f"Multi-segment Exponential R² = {fit_results['multi_segment_exponential']['r_squared']:.6f}\n")
        f.write(f"Difference: {r_squared_diff:.6f}\n")
        f.write(f"Better model based on R²: {better_model}\n")
        
        # Calculate improvement percentage
        if fit_results['single_exponential']['r_squared'] > 0:
            improvement = ((fit_results['multi_segment_exponential']['r_squared'] - 
                           fit_results['single_exponential']['r_squared']) / 
                          abs(fit_results['single_exponential']['r_squared']) * 100)
            f.write(f"Improvement with multi-segment model: {improvement:.2f}%\n")

print(f"Processing complete. All results saved to: {output_dir}")
