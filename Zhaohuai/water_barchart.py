'''

It uses the data stored in consolidated_integrated_areas_normalized.csv (or similar), which is derived from 
consolidated_integrated_areas.csv, obtained from the code curve_fitting_and_stark -> all_experiments_at_once_peak_dont_move_v2.py (final section, determined at 195.8s)



'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Preserve fonts in SVG

# Set file path and output directory
file_path = '/Users/danielsinausia/Documents/Experiments/bar charts/consolidated_integrated_areas_07_170s_normalized.csv'
output_dir = os.path.dirname(file_path)

# Load the consolidated data
df = pd.read_csv(file_path)
df = df.drop_duplicates(subset=['Experiment', 'File'], keep='first')

# Define a mapping function to categorize files
def map_file_label(file_name):
    if 'Reconstruction_based_on_CO_peak_in_eigenspectra_withoutbackground_correction_integrated_areas' in file_name:
        return 'Raw data' 
    elif 'Diffusion_layer_withoutbackground_correction_at_170s_integrated_areas' in file_name:
        return 'Diffusion'
    elif 'Interfacial_layer_withoutbackground_correction_at_170s_integrated_areas' in file_name:
        return 'Interfacial'
    elif 'mean-center contribution_withoutbackground_correction_at_170s_integrated_areas' in file_name:
        return 'MC'
    else:
        # Ignore files that do not match any of the above patterns
        return None

# Apply the mapping function and filter out rows with None in the 'Short Label' column
df['Short Label'] = df['File'].apply(map_file_label)
df = df[df['Short Label'].notna()] 

# Grouping data by 'Experiment' and 'Short Label' to remove duplicates
columns_to_aggregate = df.columns[28:34]  # Columns AC to AH
df = df.groupby(['Experiment', 'Short Label'], as_index=False)[columns_to_aggregate].sum()

# Grouping data by experiment
experiments = df['Experiment'].unique()

# Initialize color palette for hues
palette = sns.color_palette("magma", n_colors=len(columns_to_aggregate))

# Define the desired order for the short labels
short_label_order = ['Raw data', 'MC', 'Diffusion', 'Interfacial']  # Predefined label order

# Create a bar chart for each experiment
for experiment in experiments:
    # Filter data for the current experiment
    exp_data = df[df['Experiment'] == experiment]

    # Verify 'Short Label' column
    print(f"Processing Experiment: {experiment}")
    print("Unique Short Labels:", exp_data['Short Label'].unique())

    # Define the label order
    exp_data['Short Label'] = pd.Categorical(
        exp_data['Short Label'],
        categories=short_label_order,
        ordered=True
    )

    # Sort by the defined label order
    exp_data = exp_data.sort_values('Short Label')

    # Melt the DataFrame for plotting
    melted_data = exp_data.melt(
        id_vars=['Experiment', 'Short Label'], 
        value_vars=columns_to_aggregate, 
        var_name='Water structure', 
        value_name='Integrated area (a.u.)'
    )

    # Ensure numeric values in the 'Integrated area (a.u.)' column
    melted_data['Integrated area (a.u.)'] = pd.to_numeric(melted_data['Integrated area (a.u.)'], errors='coerce')

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=melted_data, 
        x='Short Label', 
        weights='Integrated area (a.u.)', 
        hue='Water structure', 
        multiple='stack', 
        palette=palette, 
        shrink=0.8
    )
    
    # Customize the chart
    plt.title(f"Experiment: {experiment}", fontsize=16)
    plt.xlabel("Files", fontsize=12)
    plt.ylabel("Integrated Area (a.u.)", fontsize=12)
    plt.legend(title='Water Structure', fontsize=10)
    plt.xticks(rotation=45)

    # Save the plot
    plot_name = f"{experiment}_stacked_bar_chart"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{plot_name}.png"))
    plt.savefig(os.path.join(output_dir, f"{plot_name}.svg"))
    plt.close()

print(f"Bar charts saved in {output_dir}.")
