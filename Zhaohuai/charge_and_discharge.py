import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import pandas as pd
import warnings

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

# Define alternating-sign multi-segment exponential function
def alternating_exp_func(x, *params):
    y = np.zeros_like(x, dtype=float)
    segment_length = 91
    
    # Each segment requires 3 parameters (a, |b|, c)
    # We will use the absolute value of b and apply the sign based on segment index
    params_per_segment = 3
    num_segments = len(params) // params_per_segment
    
    for i in range(num_segments):
        # Get the parameters for this segment
        segment_params = params[i*params_per_segment:(i+1)*params_per_segment]
        a, abs_b, c = segment_params
        
        # Determine sign of b based on segment index (alternating)
        b = abs_b * (-1)**i  # This will give alternating signs: +, -, +, -, ...
        
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

# Calculate sum of squares total (for R-squared calculations)
ss_tot = np.sum((y - np.mean(y))**2)

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

# Fit individual exponential curves to each segment (regular multi-segment fit)
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

# Fit the regular multi-segment exponential model
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
    print(f"Error during regular multi-segment exponential curve fitting: {str(e)}")
    print("Using individual segment fits as fallback for regular multi-segment")
    
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

# Fit individual exponential curves with alternating signs
alt_segment_fits = []
alt_segment_params = []
alt_segment_r_squared = []

# Determine initial sign (we'll make the first segment positive)
current_sign = 1

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
        p0 = [segment_y[0], 0.1 * current_sign, 0.0]
        
        # Set parameter bounds to enforce sign constraint
        bounds = ([0, 0.000001 if current_sign > 0 else -np.inf, -np.inf], 
                 [np.inf, np.inf if current_sign > 0 else -0.000001, np.inf])
        
        # Perform the curve fitting for this segment with sign constraint
        popt, pcov = curve_fit(exp_func, segment_x_rel, segment_y, p0=p0, bounds=bounds)
        
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
        
        print(f"Alternating segment {i+1}: Exponential fit successful with sign {'+' if b > 0 else '-'} and R-squared: {segment_r2:.4f}")
        
        # Store segment results
        alt_segment_fits.append((segment_x, segment_y_fit))
        alt_segment_params.append((a, b, c))
        alt_segment_r_squared.append(segment_r2)
        
        # Flip sign for next segment
        current_sign *= -1
        
    except Exception as e:
        print(f"Error fitting alternating segment {i+1}: {str(e)}")
        # Use a constant value as fallback
        a, b, c = np.mean(segment_y), 0.1 * current_sign, 0
        segment_y_fit = np.full_like(segment_x_rel, np.mean(segment_y))
        alt_segment_fits.append((segment_x, segment_y_fit))
        alt_segment_params.append((a, b, c))
        alt_segment_r_squared.append(0)
        
        # Flip sign for next segment
        current_sign *= -1

# Fit the alternating-sign multi-segment exponential model
try:
    # Initial parameter guess - use absolute values of the b parameters
    p0 = []
    for a, b, c in alt_segment_params:
        p0.extend([a, abs(b), c])
    
    # Perform the curve fitting
    popt, pcov = curve_fit(alternating_exp_func, x, y, p0=p0, maxfev=10000)
    
    # Calculate the fitted curve
    y_fit_alt = alternating_exp_func(x, *popt)
    
    # Calculate R-squared for alternating-sign multi-segment exponential fit
    residuals_alt = y - y_fit_alt
    ss_res_alt = np.sum(residuals_alt**2)
    r_squared_alt = 1 - (ss_res_alt / ss_tot)
    
    print(f"Alternating-sign multi-segment exponential fit successful with {num_segments} segments")
    print(f"Alternating-sign multi-segment exponential R-squared: {r_squared_alt:.4f}")
    
    # Extract the parameters
    alt_multi_segment_params = []
    for i in range(num_segments):
        if i*3+2 < len(popt):
            a, abs_b, c = popt[i*3], popt[i*3+1], popt[i*3+2]
            b = abs_b * (-1)**i  # Apply alternating sign
            alt_multi_segment_params.append({'a': a, 'b': b, 'c': c})
    
    fit_results['alternating_sign_exponential'] = {
        'r_squared': r_squared_alt,
        'y_fit': y_fit_alt,
        'residuals': residuals_alt,
        'params': alt_multi_segment_params,
        'segment_length': segment_length
    }
    
except Exception as e:
    print(f"Error during alternating-sign multi-segment exponential curve fitting: {str(e)}")
    print("Using individual alternating-sign segment fits as fallback")
    
    # Create combined fit from individual alternating-sign segments
    y_fit_alt = np.zeros_like(y)
    for i, (segment_x, segment_y_fit) in enumerate(alt_segment_fits):
        y_fit_alt[segment_x[0]:segment_x[-1]+1] = segment_y_fit
    
    # Calculate R-squared for combined individual alternating-sign fits
    residuals_alt = y - y_fit_alt
    ss_res_alt = np.sum(residuals_alt**2)
    r_squared_alt = 1 - (ss_res_alt / ss_tot)
    
    print(f"Combined individual alternating-sign segment fits R-squared: {r_squared_alt:.4f}")
    
    fit_results['alternating_sign_exponential'] = {
        'r_squared': r_squared_alt,
        'y_fit': y_fit_alt,
        'residuals': residuals_alt,
        'params': alt_segment_params,
        'segment_r_squared': alt_segment_r_squared,
        'segment_length': segment_length
    }

# Determine the best model based on R-squared
models = [model for model in fit_results if fit_results[model] is not None]
if models:
    best_model = max(models, key=lambda model: fit_results[model]['r_squared'])
    print(f"Best model based on R-squared: {best_model} (R² = {fit_results[best_model]['r_squared']:.4f})")
    
    # Calculate differences in R-squared
    for model in models:
        if model != best_model:
            r_squared_diff = fit_results[best_model]['r_squared'] - fit_results[model]['r_squared']
            print(f"  {best_model} is better than {model} by {r_squared_diff:.4f}")
else:
    print("No successful fits")

# Create comparison plot for all three models
plt.figure(figsize=(15, 10))

# Data points
plt.scatter(x, y, label='Data points', color='blue', alpha=0.5)

# Single exponential fit
if 'single_exponential' in fit_results and fit_results['single_exponential']:
    plt.plot(x, fit_results['single_exponential']['y_fit'], 'r-', 
             label=f"Single Exponential\nR² = {fit_results['single_exponential']['r_squared']:.4f}", 
             linewidth=2)

# Regular multi-segment exponential fit
if 'multi_segment_exponential' in fit_results and fit_results['multi_segment_exponential']:
    plt.plot(x, fit_results['multi_segment_exponential']['y_fit'], 'g-', 
             label=f"Multi-segment Exponential\nR² = {fit_results['multi_segment_exponential']['r_squared']:.4f}", 
             linewidth=2)

# Alternating-sign multi-segment exponential fit
if 'alternating_sign_exponential' in fit_results and fit_results['alternating_sign_exponential']:
    plt.plot(x, fit_results['alternating_sign_exponential']['y_fit'], 'm-', 
             label=f"Alternating-sign Exponential\nR² = {fit_results['alternating_sign_exponential']['r_squared']:.4f}", 
             linewidth=2)

plt.legend(fontsize=12, loc='best')
plt.title('Model Comparison: Three Exponential Fitting Approaches', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('PCA Score (Column 2)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add segment boundaries as vertical lines
for i in range(1, num_segments):
    boundary = i * segment_length
    if boundary < len(x):
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)

# Add R-squared comparison text on the plot
if len(models) > 1:
    text = "Model comparison:\n"
    for model in models:
        model_name = {
            'single_exponential': 'Single Exponential',
            'multi_segment_exponential': 'Multi-segment Exp',
            'alternating_sign_exponential': 'Alternating-sign Exp'
        }.get(model, model)
        text += f"{model_name} R² = {fit_results[model]['r_squared']:.4f}\n"
    text += f"Best model: {best_model}"
    
    plt.text(0.02, 0.98, text, 
             transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save comparison plot in both formats
comparison_plot_base = os.path.join(output_dir, 'three_model_comparison_plot')
png_path, svg_path = save_plot(plt, comparison_plot_base)
print(f"Saved three-model comparison plot to: {png_path} and {svg_path}")

# Create individual plots for each model
# Single exponential fit
if 'single_exponential' in fit_results and fit_results['single_exponential']:
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
if 'multi_segment_exponential' in fit_results and fit_results['multi_segment_exponential']:
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

# Alternating-sign multi-segment exponential fit
if 'alternating_sign_exponential' in fit_results and fit_results['alternating_sign_exponential']:
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, label='Data points', color='blue', alpha=0.7)
    plt.plot(x, fit_results['alternating_sign_exponential']['y_fit'], 'm-', 
             label=f"Alternating-sign Exponential (91-point segments)", linewidth=2)
    
    # Add vertical lines to show segment boundaries
    for i in range(1, num_segments):
        boundary = i * segment_length
        if boundary < len(x):
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    plt.legend(fontsize=12)
    plt.title('Alternating-sign Multi-segment Exponential Fit of PCA Scores', fontsize=14)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('PCA Score (Column 2)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'R² = {fit_results["alternating_sign_exponential"]["r_squared"]:.4f}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    alt_exp_base = os.path.join(output_dir, f'alternating_sign_exponential_R2_{fit_results["alternating_sign_exponential"]["r_squared"]:.4f}')
    png_path, svg_path = save_plot(plt, alt_exp_base)
    print(f"Saved alternating-sign exponential plot to: {png_path} and {svg_path}")

# Save comprehensive results to CSV
results_df = pd.DataFrame({
    'Index': x,
    'Original_Data': y,
})

if 'single_exponential' in fit_results and fit_results['single_exponential']:
    results_df['Single_Exponential_Fitted_Values'] = fit_results['single_exponential']['y_fit']
    results_df['Single_Exponential_Residuals'] = fit_results['single_exponential']['residuals']

if 'multi_segment_exponential' in fit_results and fit_results['multi_segment_exponential']:
    results_df['Multi_Segment_Exponential_Fitted_Values'] = fit_results['multi_segment_exponential']['y_fit']
    results_df['Multi_Segment_Exponential_Residuals'] = fit_results['multi_segment_exponential']['residuals']

if 'alternating_sign_exponential' in fit_results and fit_results['alternating_sign_exponential']:
    results_df['Alternating_Sign_Exponential_Fitted_Values'] = fit_results['alternating_sign_exponential']['y_fit']
    results_df['Alternating_Sign_Exponential_Residuals'] = fit_results['alternating_sign_exponential']['residuals']

csv_path = os.path.join(output_dir, 'three_model_comparison_results.csv')
results_df.to_csv(csv_path, index=False)
print(f"Saved comprehensive results to: {csv_path}")

# Save parameters and comparison to text file
params_path = os.path.join(output_dir, 'three_model_fit_comparison.txt')
with open(params_path, 'w') as f:
    f.write(f"MODEL COMPARISON: THREE EXPONENTIAL FITTING APPROACHES\n")
    f.write(f"=================================================\n\n")
    
    # Single exponential model details
    if 'single_exponential' in fit_results and fit_results['single_exponential']:
        f.write(f"SINGLE EXPONENTIAL MODEL: y = a * exp(b * x) + c\n")
        f.write(f"Parameters:\n")
        f.write(f"a = {fit_results['single_exponential']['params']['a']:.6f}\n")
        f.write(f"b = {fit_results['single_exponential']['params']['b']:.6f}\n")
        f.write(f"c = {fit_results['single_exponential']['params']['c']:.6f}\n")
        f.write(f"R-squared (R²): {fit_results['single_exponential']['r_squared']:.6f}\n")
        f.write(f"Equation: y = {fit_results['single_exponential']['equation']}\n\n")
    else:
        f.write(f"SINGLE EXPONENTIAL MODEL: Fitting failed\n\n")
    
    # Regular multi-segment exponential model details
    if 'multi_segment_exponential' in fit_results and fit_results['multi_segment_exponential']:
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
    
    # Alternating-sign multi-segment exponential model details
    if 'alternating_sign_exponential' in fit_results and fit_results['alternating_sign_exponential']:
        f.write(f"ALTERNATING-SIGN MULTI-SEGMENT EXPONENTIAL MODEL: (segments of {segment_length} points)\n")
        f.write(f"Parameters for each segment (a, b, c for y = a * exp(b * x) + c):\n")
        f.write(f"NOTE: The b parameter is forced to alternate sign between segments\n\n")
        
        # If we have segment-specific R² values, include them
        if 'segment_r_squared' in fit_results['alternating_sign_exponential']:
            for i, params in enumerate(fit_results['alternating_sign_exponential']['params']):
                if isinstance(params, tuple):
                    a, b, c = params
                    r2 = fit_results['alternating_sign_exponential']['segment_r_squared'][i]
                    sign = '+' if b > 0 else '-'
                    f.write(f"Segment {i+1}: a = {a:.6f}, b = {b:.6f} ({sign}), c = {c:.6f}, R² = {r2:.6f}\n")
                else:
                    a, b, c = params['a'], params['b'], params['c']
                    sign = '+' if b > 0 else '-'
                    f.write(f"Segment {i+1}: a = {a:.6f}, b = {b:.6f} ({sign}), c = {c:.6f}\n")
        else:
            for i, params in enumerate(fit_results['alternating_sign_exponential']['params']):
                if isinstance(params, dict):
                    a, b, c = params['a'], params['b'], params['c']
                    sign = '+' if b > 0 else '-'
                    f.write(f"Segment {i+1}: a = {a:.6f}, b = {b:.6f} ({sign}), c = {c:.6f}\n")
        
        f.write(f"\nOverall R-squared (R²): {fit_results['alternating_sign_exponential']['r_squared']:.6f}\n\n")
    else:
        f.write(f"ALTERNATING-SIGN MULTI-SEGMENT EXPONENTIAL MODEL: Fitting failed\n\n")
    
    # Model comparison
    if len(models) > 1:
        f.write(f"MODEL COMPARISON:\n")
        for model in models:
            model_name = {
                'single_exponential': 'Single Exponential',
                'multi_segment_exponential': 'Multi-segment Exponential',
                'alternating_sign_exponential': 'Alternating-sign Exponential'
            }.get(model, model)
            f.write(f"{model_name} R² = {fit_results[model]['r_squared']:.6f}\n")
        
        f.write(f"\nBest model based on R²: {best_model}\n")
        
        # Calculate improvement percentages
        best_r2 = fit_results[best_model]['r_squared']
        for model in models:
            if model != best_model:
                r2_diff = best_r2 - fit_results[model]['r_squared']
                f.write(f"{best_model} is better than {model} by {r2_diff:.6f} (R² difference)\n")

# Create a plot showing the slope (b parameter) for each segment in both multi-segment models
if ('multi_segment_exponential' in fit_results and fit_results['multi_segment_exponential'] and 
    'alternating_sign_exponential' in fit_results and fit_results['alternating_sign_exponential']):
    
    plt.figure(figsize=(12, 8))
    
    # Extract b parameters for regular multi-segment model
    regular_b_params = []
    if isinstance(fit_results['multi_segment_exponential']['params'][0], tuple):
        for a, b, c in fit_results['multi_segment_exponential']['params']:
            regular_b_params.append(b)
    else:
        for params in fit_results['multi_segment_exponential']['params']:
            regular_b_params.append(params['b'])
    
    # Extract b parameters for alternating-sign model
    alt_b_params = []
    if isinstance(fit_results['alternating_sign_exponential']['params'][0], tuple):
        for a, b, c in fit_results['alternating_sign_exponential']['params']:
            alt_b_params.append(b)
    else:
        for params in fit_results['alternating_sign_exponential']['params']:
            alt_b_params.append(params['b'])
    
    # Create segment indices
    segment_indices = range(1, len(regular_b_params) + 1)
    
    # Plot the b parameters
    plt.plot(segment_indices, regular_b_params, 'g-o', label='Regular Multi-segment', linewidth=2, markersize=8)
    plt.plot(segment_indices, alt_b_params, 'm-o', label='Alternating-sign', linewidth=2, markersize=8)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.legend(fontsize=12)
    plt.title('Comparison of Exponential Slope (b parameter) Between Models', fontsize=14)
    plt.xlabel('Segment Number', fontsize=12)
    plt.ylabel('Slope (b parameter)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(segment_indices)
    
    # Save the slope comparison plot
    slope_plot_base = os.path.join(output_dir, 'slope_parameter_comparison')
    png_path, svg_path = save_plot(plt, slope_plot_base)
    print(f"Saved slope parameter comparison plot to: {png_path} and {svg_path}")

print(f"Processing complete. All results saved to: {output_dir}")
