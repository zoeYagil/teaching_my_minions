'''

directory where you want to save your results in output_dir. 
Files to use in:
def main():
    """Main function to run the multi-dataset pentagon analysis"""
    
    print("Multi-Dataset Pentagon Analysis Script")
    print("=" * 60)
    
    # CONFIGURE YOUR CSV FILES AND LABELS HERE
    # Format: (file_path, label_for_legend)
    csv_files_with_labels = [
        ("/Users/danielsinausia/Downloads/PCA_exp_fit_results/exponential_terms_analysis.csv", "Dataset 1"),
        ("/Users/danielsinausia/Downloads/PCA_exp_fit_results/exponential_terms_analysis.csv", "Dataset 2")


'''




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from typing import Dict, List, Tuple, Optional

# Define output directory
output_dir = "/Users/danielsinausia/Downloads/PCA_exp_fit_results"

def load_and_analyze_data(csv_path: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Load CSV data and extract the required values for pentagon plot"""
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV '{os.path.basename(csv_path)}' with {len(df)} rows")
        
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
        print(f"\nPentagon Corner Values for '{os.path.basename(csv_path)}':")
        print(f"Corner 1 (Avg Even Exp Term): {results['corner_1']:.6f}")
        print(f"Corner 2 (Avg Odd Exp Term): {results['corner_2']:.6f}")
        print(f"Corner 3 (Exp Term Ratio): {results['corner_3']:.6f}")
        print(f"Corner 4 (Max Mean 3360 Diff): {results['corner_4']:.6f}")
        print(f"Corner 5 (Segment Ratio): {results['corner_5']:.6f}")
        
        return results, segment_data
        
    except Exception as e:
        print(f"Error loading or analyzing data from '{csv_path}': {str(e)}")
        return None, None

def create_multi_pentagon_plot(datasets: Dict[str, Dict], output_path: str):
    """Create a pentagon radar plot with multiple overlaid datasets"""
    
    # Pentagon corner labels
    labels = [
        "Average Exponential Term (high overpotential",
        "Average Exponential Term (low overpotential", 
        "Abs Max Exp Term High Overpotential - Abs Max Exp Term Low Overpotential)/\nAbs Max Exp Term Overall",
        "Difference Intensity Peak 3360 Max High - MLow Overpotential",
        "Ratio between difference of exponential terms \n(last - first high overpotential) and the intensities \n of peak at 3360 (last - first high overpotential)"
    ]
    
    # Colors for different datasets
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Calculate global scaling
    all_values = []
    for data in datasets.values():
        corner_values = [
            data['corner_1'], data['corner_2'], data['corner_3'],
            data['corner_4'], data['corner_5']
        ]
        all_values.extend([abs(val) for val in corner_values])
    
    max_abs_val = max(all_values) if all_values else 1.0
    
    # Number of variables
    N = len(labels)
    
    # Calculate angles for each corner (starting from top, going clockwise)
    angles = [np.pi/2 - (2 * np.pi * i / N) for i in range(N)]
    
    # Create regular plot (not polar)
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw reference pentagons (concentric)
    pentagon_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    for level in pentagon_levels:
        # Calculate pentagon vertices for this level
        pentagon_x = [level * np.cos(angle) for angle in angles]
        pentagon_y = [level * np.sin(angle) for angle in angles]
        pentagon_x.append(pentagon_x[0])  # Close the pentagon
        pentagon_y.append(pentagon_y[0])
        
        # Draw pentagon outline
        ax.plot(pentagon_x, pentagon_y, 'gray', linewidth=0.5, alpha=0.3)
    
    # Draw radial lines from center to each corner
    for angle in angles:
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 'gray', linewidth=0.5, alpha=0.3)
    
    # Plot each dataset
    legend_elements = []
    
    for idx, (label, values) in enumerate(datasets.items()):
        color = colors[idx % len(colors)]
        
        # Extract values
        corner_values = [
            values['corner_1'], values['corner_2'], values['corner_3'],
            values['corner_4'], values['corner_5']
        ]
        
        # Convert to absolute values and normalize
        abs_values = [abs(val) for val in corner_values]
        
        # Scale values to a reasonable range (0.1 to 1.0 for visibility)
        normalized_values = []
        for val in abs_values:
            if val == 0:
                normalized_values.append(0.1)  # Minimum visible value
            else:
                normalized_values.append(0.1 + 0.9 * (val / max_abs_val))
        
        # Calculate data points
        data_x = [normalized_values[i] * np.cos(angles[i]) for i in range(N)]
        data_y = [normalized_values[i] * np.sin(angles[i]) for i in range(N)]
        data_x.append(data_x[0])  # Close the shape
        data_y.append(data_y[0])
        
        # Plot the data as a filled pentagon
        line = ax.plot(data_x, data_y, 'o-', linewidth=2.5, color=color, 
                      markersize=6, alpha=0.8, label=label)
        ax.fill(data_x, data_y, alpha=0.15, color=color)
        
        legend_elements.append(line[0])
    
    # Add corner labels (only once, outside the loop)
    for i, (angle, label_text) in enumerate(zip(angles, labels)):
        # Position labels outside the plot
        label_distance = 1.25
        label_x = label_distance * np.cos(angle)
        label_y = label_distance * np.sin(angle)
        
        ax.text(label_x, label_y, label_text, ha='center', va='center', 
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
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
            scale_x = level * np.cos(angles[0]) * 0.7  # Slightly inside to avoid overlap
            scale_y = level * np.sin(angles[0]) * 0.7
            
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
    plt.title('Multi-Dataset Pentagon Analysis Plot\nExponential Terms and Mean 3360 Relationships', 
              fontsize=16, fontweight='bold', pad=30)
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
              frameon=True, fancybox=True, shadow=True)
    
    # Add scaling info
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
    
    print(f"Multi-pentagon plot saved to: {png_path} and {svg_path}")
    
    return png_path, svg_path

def analyze_multiple_csvs(csv_files_with_labels: List[Tuple[str, str]]) -> Dict[str, Dict]:
    """
    Analyze multiple CSV files and return their pentagon data
    
    Args:
        csv_files_with_labels: List of tuples (csv_file_path, label)
    
    Returns:
        Dictionary mapping labels to their pentagon corner values
    """
    datasets = {}
    
    for csv_path, label in csv_files_with_labels:
        if not os.path.exists(csv_path):
            print(f"Warning: File not found: {csv_path}")
            continue
        
        results, _ = load_and_analyze_data(csv_path)
        if results is not None:
            datasets[label] = results
        else:
            print(f"Failed to analyze {csv_path}")
    
    return datasets

def save_combined_results(datasets: Dict[str, Dict], output_path: str):
    """Save detailed results for all datasets to text file"""
    
    with open(output_path, 'w') as f:
        f.write("MULTI-DATASET PENTAGON ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        for label, results in datasets.items():
            f.write(f"DATASET: {label}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Exponential Term (high overpotential): {results['corner_1']:.6f}\n")
            f.write(f"Average Exponential Term (low overpotential): {results['corner_2']:.6f}\n")
            f.write(f"(Abs Max Exp Term High Overpotential - Abs Max Exp Term Low Overpotential)/Abs Max Exp Term Overall: {results['corner_3']:.6f}\n")
            f.write(f"Difference Intensity Peak 3360 Max High - MLow Overpotential: {results['corner_4']:.6f}\n")
            f.write(f"Ratio between difference of exponential terms (last - first high overpotential) and the intensities of peak at 3360 (last - first high overpotential)): {results['corner_5']:.6f}\n\n")
        
        f.write("CALCULATION DETAILS:\n")
        f.write("-" * 40 + "\n")
        f.write("Corner 1: Average of Regular_Multi_Segment_Exponential_Term for even segments\n")
        f.write("Corner 2: Average of Regular_Multi_Segment_Exponential_Term for odd segments\n")
        f.write("Corner 3: (abs(max_even) - abs(max_odd))/abs(overall_max) for exponential terms\n")
        f.write("Corner 4: (Highest Max_Abs_Mean_3360 for even) - (Highest Max_Abs_Mean_3360 for odd)\n")
        f.write("Corner 5: (Segment 10 - Segment 2 exponential term) / (Segment 10 - Segment 2 Mean 3360)\n")

def main():
    """Main function to run the multi-dataset pentagon analysis"""
    
    print("Multi-Dataset Pentagon Analysis Script")
    print("=" * 60)
    
    # CONFIGURE YOUR CSV FILES AND LABELS HERE
    # Format: (file_path, label_for_legend)
    csv_files_with_labels = [
        ("/Users/danielsinausia/Downloads/PCA_exp_fit_results/exponential_terms_analysis.csv", "Dataset 1"),
        ("/Users/danielsinausia/Downloads/PCA_exp_fit_results/exponential_terms_analysis.csv", "Dataset 2")
        # Add more files here like:
        # ("/path/to/second_file.csv", "Dataset 2"),
        # ("/path/to/third_file.csv", "Dataset 3"),
    ]
    
    # You can also add files programmatically:
    # csv_files_with_labels.extend([
    #     ("/path/to/another_file.csv", "Another Dataset"),
    # ])
    
    if len(csv_files_with_labels) == 0:
        print("No CSV files configured. Please add file paths and labels to the csv_files_with_labels list.")
        return
    
    # Analyze all datasets
    datasets = analyze_multiple_csvs(csv_files_with_labels)
    
    if not datasets:
        print("No valid datasets found. Exiting.")
        return
    
    print(f"\nSuccessfully analyzed {len(datasets)} datasets")
    
    # Create multi-pentagon plot
    plot_base_path = os.path.join(output_dir, 'multi_pentagon_analysis_plot')
    png_path, svg_path = create_multi_pentagon_plot(datasets, plot_base_path)
    
    # Save detailed results to text file
    results_txt_path = os.path.join(output_dir, 'multi_pentagon_analysis_results.txt')
    save_combined_results(datasets, results_txt_path)
    
    print(f"\nDetailed results saved to: {results_txt_path}")
    print("Multi-dataset pentagon analysis complete!")

# Alternative function for easy single-file usage
def analyze_single_file(csv_path: str, label: str = None):
    """Convenience function to analyze a single file (backward compatibility)"""
    if label is None:
        label = os.path.basename(csv_path)
    
    csv_files_with_labels = [(csv_path, label)]
    datasets = analyze_multiple_csvs(csv_files_with_labels)
    
    if datasets:
        plot_base_path = os.path.join(output_dir, f'pentagon_analysis_{label.replace(" ", "_")}')
        create_multi_pentagon_plot(datasets, plot_base_path)

if __name__ == "__main__":
    main()
