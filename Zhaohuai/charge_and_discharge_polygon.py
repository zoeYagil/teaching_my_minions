import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

# Define file paths
csv_file_path = "/Users/danielsinausia/Downloads/PCA_exp_fit_results/exponential_terms_analysis.csv"
output_dir = "/Users/danielsinausia/Downloads/PCA_exp_fit_results"

def load_and_analyze_data(csv_path):
    """Load CSV data and extract the required values for pentagon plot"""
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        
        # Filter out summary rows and get only the actual segment data
        segment_data = df[df['Segment_Number'].apply(lambda x: str(x).isdigit())].copy()
        segment_data['Segment_Number'] = segment_data['Segment_Number'].astype(int)
        
        print(f"Found {len(segment_data)} actual segments")
        
        # 1. Corner 1: Average even from Regular_Multi_Segment_Exponential_Term
        even_segments = segment_data[segment_data['Segment_Type'] == 'Even']
        avg_even_exp_term = even_segments['Regular_Multi_Segment_Exponential_Term'].mean()
        
        # 2. Corner 2: Average odd from Regular_Multi_Segment_Exponential_Term
        odd_segments = segment_data[segment_data['Segment_Type'] == 'Odd']
        avg_odd_exp_term = odd_segments['Regular_Multi_Segment_Exponential_Term'].mean()
        
        # 3. Corner 3: Ratio from the RATIO_CALC row
        ratio_row = df[df['Segment_Number'] == 'RATIO_CALC']
        if not ratio_row.empty:
            # Extract the ratio value from the equation string or from the exponential term column
            ratio_exp_term = ratio_row['Regular_Multi_Segment_Exponential_Term'].iloc[0]
        else:
            # Calculate manually if ratio row not found
            all_exp_terms = segment_data['Regular_Multi_Segment_Exponential_Term'].dropna()
            max_abs_even = even_segments['Regular_Multi_Segment_Exponential_Term'].abs().max()
            max_abs_odd = odd_segments['Regular_Multi_Segment_Exponential_Term'].abs().max()
            overall_max_abs = all_exp_terms.abs().max()
            ratio_exp_term = (max_abs_even - max_abs_odd) / overall_max_abs if overall_max_abs != 0 else 0
        
        # 4. Corner 4: Highest Max_Abs_Mean_3360 for even - highest for odd
        max_even_mean_3360 = even_segments['Max_Abs_Mean_3360'].max()
        max_odd_mean_3360 = odd_segments['Max_Abs_Mean_3360'].max()
        diff_max_mean_3360 = max_even_mean_3360 - max_odd_mean_3360
        
        # 5. Corner 5: (Segment 10 - Segment 2 in exp term) / (Segment 10 - Segment 2 in Mean_3360)
        segment_2 = segment_data[segment_data['Segment_Number'] == 2]
        segment_10 = segment_data[segment_data['Segment_Number'] == 10]
        
        if not segment_2.empty and not segment_10.empty:
            exp_term_2 = segment_2['Regular_Multi_Segment_Exponential_Term'].iloc[0]
            exp_term_10 = segment_10['Regular_Multi_Segment_Exponential_Term'].iloc[0]
            mean_3360_2 = segment_2['Max_Abs_Mean_3360'].iloc[0]
            mean_3360_10 = segment_10['Max_Abs_Mean_3360'].iloc[0]
            
            exp_diff = exp_term_10 - exp_term_2
            mean_3360_diff = mean_3360_10 - mean_3360_2
            
            corner_5_ratio = exp_diff / mean_3360_diff if mean_3360_diff != 0 else 0
        else:
            corner_5_ratio = 0
            print("Warning: Segment 2 or 10 not found")
        
        # Create results dictionary
        results = {
            'corner_1': avg_even_exp_term,
            'corner_2': avg_odd_exp_term,
            'corner_3': ratio_exp_term,
            'corner_4': diff_max_mean_3360,
            'corner_5': corner_5_ratio
        }
        
        # Print results
        print("\nPentagon Corner Values:")
        print(f"Corner 1 (Avg Even Exp Term): {results['corner_1']:.6f}")
        print(f"Corner 2 (Avg Odd Exp Term): {results['corner_2']:.6f}")
        print(f"Corner 3 (Exp Term Ratio): {results['corner_3']:.6f}")
        print(f"Corner 4 (Max Mean 3360 Diff): {results['corner_4']:.6f}")
        print(f"Corner 5 (Segment Ratio): {results['corner_5']:.6f}")
        
        return results, segment_data
        
    except Exception as e:
        print(f"Error loading or analyzing data: {str(e)}")
        return None, None

def create_pentagon_plot(values, output_path):
    """Create a pentagon radar plot with the given values - pentagons only, no circles"""
    
    # Pentagon corner labels
    labels = [
        "Corner 1\n(Avg Even Exp Term)",
        "Corner 2\n(Avg Odd Exp Term)", 
        "Corner 3\n(Exp Term Ratio)",
        "Corner 4\n(Max Mean 3360 Diff)",
        "Corner 5\n(Segment Ratio)"
    ]
    
    # Extract values
    corner_values = [
        values['corner_1'],
        values['corner_2'],
        values['corner_3'],
        values['corner_4'],
        values['corner_5']
    ]
    
    # Convert to absolute values and scale appropriately
    abs_values = [abs(val) for val in corner_values]
    
    # Handle case where all values might be very small or zero
    max_abs_val = max(abs_values)
    if max_abs_val == 0:
        normalized_values = [0.1] * 5  # Small but visible values
        print("Warning: All values are zero, using small default values for visibility")
    else:
        # Scale values to a reasonable range (0.1 to 1.0 for visibility)
        normalized_values = []
        for val in abs_values:
            if val == 0:
                normalized_values.append(0.1)  # Minimum visible value
            else:
                normalized_values.append(0.1 + 0.9 * (val / max_abs_val))
    
    # Number of variables
    N = len(labels)
    
    # Calculate angles for each corner (starting from top, going clockwise)
    # For pentagon: 0° at top, then 72° intervals
    angles = [np.pi/2 - (2 * np.pi * i / N) for i in range(N)]
    
    # Create regular plot (not polar)
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw reference pentagons (concentric)
    pentagon_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    for level in pentagon_levels:
        # Calculate pentagon vertices for this level
        pentagon_x = [level * np.cos(angle) for angle in angles]
        pentagon_y = [level * np.sin(angle) for angle in angles]
        pentagon_x.append(pentagon_x[0])  # Close the pentagon
        pentagon_y.append(pentagon_y[0])
        
        # Draw pentagon outline
        ax.plot(pentagon_x, pentagon_y, 'gray', linewidth=0.5, alpha=0.5)
    
    # Draw radial lines from center to each corner
    for angle in angles:
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 'gray', linewidth=0.5, alpha=0.5)
    
    # Calculate data points
    data_x = [normalized_values[i] * np.cos(angles[i]) for i in range(N)]
    data_y = [normalized_values[i] * np.sin(angles[i]) for i in range(N)]
    data_x.append(data_x[0])  # Close the shape
    data_y.append(data_y[0])
    
    # Plot the data as a filled pentagon
    ax.plot(data_x, data_y, 'o-', linewidth=3, color='blue', markersize=8)
    ax.fill(data_x, data_y, alpha=0.25, color='lightblue')
    
    # Add labels and values
    for i, (angle, label, value) in enumerate(zip(angles, labels, corner_values)):
        # Position labels outside the plot
        label_distance = 1.3
        label_x = label_distance * np.cos(angle)
        label_y = label_distance * np.sin(angle)
        
        ax.text(label_x, label_y, label, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Add value below the label
        value_distance = 1.45
        value_x = value_distance * np.cos(angle)
        value_y = value_distance * np.sin(angle)
        
        ax.text(value_x, value_y, f'{value:.4f}', ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    
    # Add concentric pentagon labels showing scale
    if max_abs_val > 0:
        for i, level in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
            # Convert normalized value back to original scale
            if level <= 0.1:
                original_val = 0
            else:
                original_val = ((level - 0.1) / 0.9) * max_abs_val
            
            # Place scale labels along one of the radial lines
            scale_x = level * np.cos(angles[0]) * 0.8  # Slightly inside to avoid overlap
            scale_y = level * np.sin(angles[0]) * 0.8
            
            ax.text(scale_x, scale_y, f'{original_val:.3f}', 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow', alpha=0.7))
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add title
    plt.title('Pentagon Analysis Plot\nExponential Terms and Mean 3360 Relationships', 
              fontsize=16, fontweight='bold', pad=30)
    
    # Add legend explaining the scaling
    legend_text = f"Values scaled for visibility\nOriginal range: 0 to {max_abs_val:.4f}\nAll negative values shown as absolute values"
    plt.figtext(0.02, 0.02, legend_text, fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    
    # Save as PNG
    png_path = f"{output_path}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as SVG
    svg_path = f"{output_path}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    print(f"Pentagon plot saved to: {png_path} and {svg_path}")
    
    return png_path, svg_path


def main():
    """Main function to run the pentagon analysis"""
    
    print("Pentagon Analysis Script")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: Input file not found: {csv_file_path}")
        return
    
    # Load and analyze data
    results, segment_data = load_and_analyze_data(csv_file_path)
    
    if results is None:
        print("Failed to analyze data. Exiting.")
        return
    
    # Create pentagon plot
    plot_base_path = os.path.join(output_dir, 'pentagon_analysis_plot')
    png_path, svg_path = create_pentagon_plot(results, plot_base_path)
    
    # Save detailed results to text file
    results_txt_path = os.path.join(output_dir, 'pentagon_analysis_results.txt')
    with open(results_txt_path, 'w') as f:
        f.write("PENTAGON ANALYSIS RESULTS\\n")
        f.write("=" * 50 + "\\n\\n")
        
        f.write("Corner Values:\\n")
        f.write(f"Corner 1 (Average Even Exponential Term): {results['corner_1']:.6f}\\n")
        f.write(f"Corner 2 (Average Odd Exponential Term): {results['corner_2']:.6f}\\n")
        f.write(f"Corner 3 (Exponential Term Ratio): {results['corner_3']:.6f}\\n")
        f.write(f"Corner 4 (Max Mean 3360 Difference): {results['corner_4']:.6f}\\n")
        f.write(f"Corner 5 (Segment Ratio): {results['corner_5']:.6f}\\n\\n")
        
        f.write("Calculation Details:\\n")
        f.write("Corner 1: Average of Regular_Multi_Segment_Exponential_Term for even segments\\n")
        f.write("Corner 2: Average of Regular_Multi_Segment_Exponential_Term for odd segments\\n")
        f.write("Corner 3: (abs(max_even) - abs(max_odd))/abs(overall_max) for exponential terms\\n")
        f.write("Corner 4: (Highest Max_Abs_Mean_3360 for even) - (Highest Max_Abs_Mean_3360 for odd)\\n")
        f.write("Corner 5: (Segment 10 - Segment 2 exponential term) / (Segment 10 - Segment 2 Mean 3360)\\n")
    
    print(f"\\nDetailed results saved to: {results_txt_path}")
    print("Pentagon analysis complete!")

if __name__ == "__main__":
    main()
