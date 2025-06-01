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
    """Create a pentagon radar plot with the given values"""
    
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
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add first value to end to complete the polygon
    normalized_values += normalized_values[:1]
    corner_values += corner_values[:1]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Draw the reference grid (concentric pentagons)
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles, [r]*len(angles), 'gray', linewidth=0.5, alpha=0.5)
    
    # Draw radial lines
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], 'gray', linewidth=0.5, alpha=0.5)
    
    # Plot the data
    ax.plot(angles, normalized_values, 'o-', linewidth=3, label='Data', color='blue', markersize=8)
    ax.fill(angles, normalized_values, alpha=0.25, color='lightblue')
    
    # Add labels and values
    for i, (angle, label, value) in enumerate(zip(angles[:-1], labels, corner_values[:-1])):
        # Position labels outside the plot
        ax.text(angle, 1.15, label, ha='center', va='center', fontsize=11, fontweight='bold')
        # Add value below the label
        ax.text(angle, 1.25, f'{value:.4f}', ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Configure the plot
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Create custom r-tick labels showing the actual value ranges
    if max_abs_val > 0:
        rtick_labels = []
        for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
            # Convert normalized value back to original scale
            if r <= 0.1:
                original_val = 0
            else:
                original_val = ((r - 0.1) / 0.9) * max_abs_val
            rtick_labels.append(f'{original_val:.3f}')
        ax.set_yticklabels(rtick_labels, fontsize=9)
    
    # Remove theta tick labels
    ax.set_xticklabels([])
    
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
