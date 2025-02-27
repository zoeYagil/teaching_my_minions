'''
two parts. First section does deconvolution and integration. Second one extracts integrated values of 
the peaks that you choose at a certain time to use later in the water bar charts

•	The script looks for specific files in folders listed in include_folders.
•	Files matching suffixes (_07, _08, _09) are identified and processed.
•	Additionally, specific patterns (ReconstructedData_PCs1.csv, etc.) in subfolders are considered.
•	File Validation:
•	Files starting with "DS" are processed differently from those starting with "Reconstructed".
•	The code checks for the presence of required files (alternating_file, patterns) and processes them if they exist.


I added last minute the following code, if there are errors, check if it works without it:
    
    html_path = os.path.join(folder_path, f"{filename}.html")
    html_content = mpld3.fig_to_html(plt.gcf())  # Convert Matplotlib figure to HTML
    with open(html_path, "w") as html_file:
        html_file.write(html_content)


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.integrate import trapz
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import os
import glob
import mpld3
from matplotlib.cm import magma
import matplotlib as mpl
mpl.use('SVG')

# Set font settings for better compatibility with SVG text
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths

base_dir = "/Users/danielsinausia/Documents/Experiments/"
include_folders = {"DS_00132", "DS_00133", "DS_00134", "DS_00127", "DS_00163", "DS_00131",
                       "DS_00138", "DS_00135", "DS_00139", "DS_00136", "DS_00140", "DS_00137",
                       "DS_00141", "DS_00144", "DS_00142", "DS_00145", "DS_00143", "DS_00146",
                       "DS_00181", "DS_00180", "DS_00148", "DS_00152", "DS_00149", "DS_00153"}







##########################################################
##########################################################
##########################################################
#%% NOT TO TOUCH THINGS BELOW THIS LINE
##########################################################
##########################################################
##########################################################

r2_results = []
conditions = {
    "start_reciprocal_cm": 1101,
    "end_reciprocal_cm": 3999,
    "start_reciprocal_cm_bkg": 2500,
    "end_reciprocal_cm_bkg": 3997,
    "spectrum_to_plot_as_example": 170,
    "liquid": "H2O"
}

def process_spectrum_file(file_path, start_reciprocal_cm=1101, 
                          end_reciprocal_cm=3999,
                          start_reciprocal_cm_bkg=2500, 
                          end_reciprocal_cm_bkg=3997,
                          spectrum_to_plot_as_example=600, 
                          experiment_classification='_08',
                          #experiment_classification = folder_name[-3:]
                          liquid='H2O'):
    """
    Processes a single spectrum file by performing background correction, Gaussian fitting,
    integration of peaks, and visualization.

    Args:
        file_path (str): Path to the CSV file to process.
        start_reciprocal_cm (int): Start reciprocal cm value for the spectrum range.
        end_reciprocal_cm (int): End reciprocal cm value for the spectrum range.
        start_reciprocal_cm_bkg (int): Start reciprocal cm value for background correction.
        end_reciprocal_cm_bkg (int): End reciprocal cm value for background correction.
        spectrum_to_plot_as_example (int): Index of the spectrum to plot.
        experiment_classification (str): Experiment classification for labeling plots.
        liquid (str): Type of liquid used in the experiment ('H2O' or 'D2O').

    Returns:
        None
    """

    folder = os.path.dirname(file_path)
    folder_name = os.path.basename(folder)
    
        
    if liquid == 'D2O':
        
        initial_amplitudes = [1,
                              1,
                                  1,
                                  1,
                                  1,
                                      1,
                                          1,
                                          1,
                                              1,
                                                    1,
                                                    1,
                                                        1,
                                                        1,
                                                            1,
                                                            1,
                                                                1,
                                                                1,
                                                                    1,
                                                                    1,
                                                                        1,
                                                                        1,
                                                                            1,
                                                                            1,
                                                                            1
                                                   ] #Peak positions obtained from 10.1021/jacsau.1c00281. The choice of 1 and 10 for amplitude and std dev are arbitrary, chosen with the help of chatGPT, but since I am interested in the evolution of the peaks rather than the absolute numbers, I'll go with it
        initial_params = initial_amplitudes# Define mean and sigma values for Gaussian components
        
        mean_values = [3400, 
                       2950,
                       2658,
                       2600,
                             2555,
                             2470,
                             2390,
                             2367, 
                            
                                         2320,
                                         2290,
                                         2071, # HFB CO \cite{an2023}
                                         2050, # LFB CO \cite{an2023}
                                         2000,
                                                     1860,  # CO bridge \cite{chou2020a}
                                                     1700, # any carbonyl intermediate 
                                                     1625,  # H-O-H \cite{park2019}
                                                     1560, # asym stretct (OCO) from formate \cite{moradzaman2020, hsu2022}
                                                     1508, # C=O stretch of bidentate CO3* \cite{katayama2019}
                                                         1460, # H-O-D \cite{park2019, litvak2018}
                                                         1430, # CO3 2- \cite{deng2021}
                                                         1365, # sym stretct(OCO) from formate \cite{moradzaman2020, hayden1983, hsu2022}, or HCO3- aq (\cite{deng2021})
                                                         1310, # formate
                                                             1204,  # This is D-O-D \cite{park2019, litvak2018}
                                                                    1140   # HCO strecth, computational result from \cite{katayama2019} (not calculated for D2O, but it´s at lower than H2O)
                                                                    
                                                                    #2085 missing D2O, 2050 missing H2O, 1860 moves to 1800, 1560 moves to 1541, 1625 moves to 1610 or 1639, 1460 is missing in H2O and check the last ones. the last three peaks on each are different, but it may be that it gets tricky there
                             ]
        
        
        sigma_values = [100, 
                        80,
                        35,
                        30,
                            70,
                            40,
                            50,
                            10,
                                        10,
                                        75,
                                        10,
                                        20,
                                        30,
                                                    80,
                                                    30,
                                                    23,
                                                    30,
                                                    30,
                                                        20,
                                                        30,
                                                        10,
                                                        25,
                                                            23,
                                                                10
                                          ]
    
    if liquid == 'H2O':
        
         initial_amplitudes = [1,
                              1,
                                  1,
                                  1,
                                  1,
                                      1,
                                      1,
                                          1,
                                          1,
                                              1,
                                                    1,
                                                    1,
                                                        1,
                                                        1,
                                                        1,
                                                            1,
                                                            1,
                                                            1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,1,1,1,1,1,1,1
                                                                
                                                   ] #Peak positions obtained from 10.1021/jacsau.1c00281. The choice of 1 and 10 for amplitude and std dev are arbitrary, chosen with the help of chatGPT, but since I am interested in the evolution of the peaks rather than the absolute numbers, I'll go with it
         initial_params = initial_amplitudes# Define mean and sigma values for Gaussian components
        
         mean_values = [3680, # CO2 sym and assym strect combination
                       3520,
                            3360, 
                            3210,
                            3100,
                                2870,# CH  stretch \cite{moradzaman2020} or bending CH+ stretch OCO \cite{moradzaman2020}
                                2800,
                                    2367, # CO2 \cite{winkler2022} assym strect
                                    2350, # CO2 (aq)
                                    2335,
                                    2320, # CO2 \cite{winkler2022} assym strect
                                        2127, # libration + bending \cite{verma2018, jiachen2022}
                                        #2100,
                                            2084, # HFB CO \cite{an2023}
                                            2077, # LFB CO \cite{an2023}
                                            2070,2055,
                                            1836,
                                                1800, # CO bridge \cite{chou2020a}
                                                1700, # any carbonyl intermediate 
                                                1639, # H-O-H \cite{park2019}
                                                    1610, # C=O from formate
                                                    1541, # asym stretct (OCO) from formate \cite{moradzaman2020, hsu2022}
                                                    1508, # C=O stretch of bidentate CO3* \cite{katayama2019}
                                                        1430, # CO3 2- \cite{deng2021}
                                                        1400,
                                                        1368, # sym stretct(OCO) from formate \cite{moradzaman2020, hayden1983, hsu2022}, or HCO3- aq (\cite{deng2021})
                                                        1270, # OCH bend from HCO* or bidentate HCOO* \cite{katayama2019}
                                                        1227, # Silicon dioxide
                                                        1218,
                                                        1160,1155,
                                                        1110, # HCO strecth, computational result from \cite{katayama2019}
                                                
                                          ]
            
            
         sigma_values = [70, 
                        100,
                            100,
                            80,
                            100,
                                80,
                                80,
                                    10,
                                    10,
                                    3,
                                    10,
                                        100,
                                    
                                            3,
                                            3,
                                            5,
                                            20,
                                            6,
                                                100,
                                                30,
                                                40, 
                                                    40,
                                                    40,
                                                    40,
                                                        30,
                                                        20,
                                                        40,
                                                        40,
                                                        15,
                                                        15,
                                                        10,40,
                                                        15, 
                                                    
                                          ]
        
    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))
    
    def combined_function(x, *amps):
        return sum(gaussian(x, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values))

    

    
    file_name = os.path.basename(file_path)
    
    if file_name.startswith("DS") or file_name.startswith("CZ"):
        # Process for files starting with "DS"
        df = pd.read_csv(file_path, header=None, skiprows=1)  # Original 955, others 0
        df = df.iloc[:, :]  # Activate for original
        df = df[::-1]  # Activate for original
        reciprocal_cm = df.iloc[:, 0]  # First column as wavenumbers
    elif file_name.startswith("Reconstructed"):
        # Process for other files
        df = pd.read_csv(file_path, header=None, skiprows=0)  # Original 955, others 0
        # df = df.iloc[:, :911]  # Activate for original
        # df = df[::-1]  # Activate for original
        reciprocal_cm = df.iloc[:, 0]  # First column as wavenumbers
    


    num_peaks = len(mean_values)
    labels = [mean_values[i] for i in range(num_peaks)]
    colors = magma(np.linspace(0, 1, num_peaks))
    folder_name = f"{start_reciprocal_cm}_to_{end_reciprocal_cm}"
    os.makedirs(folder_name, exist_ok=True)

    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]

    experiment_number = spectrum_to_plot_as_example
    trimmed_wavenumbers = df.iloc[start_index:end_index+1, 0].values
    spectrum = df.iloc[start_index:end_index+1, experiment_number].values

    index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
    index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
    slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
    

    intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
    background_array = slope * trimmed_wavenumbers + intercept
    spectrum_corrected = spectrum - background_array
    popt, _ = curve_fit(combined_function, trimmed_wavenumbers, spectrum_corrected, p0=initial_params, maxfev=10000)
    amps = popt
    gaussians = [gaussian(trimmed_wavenumbers, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values)]
    fitted_curve = combined_function(trimmed_wavenumbers, *popt)
    correlation_coefficient, _ = pearsonr(spectrum_corrected, fitted_curve)
    r_squared = correlation_coefficient ** 2
    print(f"R-squared: {r_squared}")   
    
    r2_results.append({
        'Experiment': os.path.basename(folder),  # Folder name as experiment identifier
        'Folder': folder,
        'File': os.path.basename(file_path),
        'R_squared': r_squared
    })
                              
    fig, ax = plt.subplots(figsize=(18, 8))
    for i, (gaussian_component, label, color) in enumerate(zip(gaussians, labels, colors)):
        plt.plot(reciprocal_cm[start_index:end_index+1], gaussian_component, label=label, color=color)
        plt.fill_between(reciprocal_cm[start_index:end_index+1], gaussian_component, color=color, alpha=0.6)
    plt.plot(reciprocal_cm[start_index:end_index+1], spectrum_corrected, label='Original Data')
    plt.plot(reciprocal_cm[start_index:end_index+1], fitted_curve, label='Fitted Curve')
    plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize=18)
    plt.ylabel('Intensity (a.u.)', fontsize=18)
    plt.xlim(start_reciprocal_cm, end_reciprocal_cm)
    plt.gca().invert_xaxis()
    plt.legend(fontsize=18, ncol=7)
    plt.title('Raw data', fontsize=18)

    folder = os.path.dirname(file_path)
    folder_path = os.path.join(folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_withoutbackground_correction"

    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")

    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    
    html_path = os.path.join(folder_path, f"{filename}.html")
    html_content = mpld3.fig_to_html(plt.gcf())  # Convert Matplotlib figure to HTML
    with open(html_path, "w") as html_file:
        html_file.write(html_content)
    plt.show()

    fitted_values = combined_function(reciprocal_cm[start_index:end_index+1], *popt)
    fit_df = pd.DataFrame({'Reciprocal_cm': reciprocal_cm[start_index:end_index+1], 'Fitted_Values': fitted_values})
    for i, (gaussian_component, label) in enumerate(zip(gaussians, labels)):
        fit_df[f'Gaussian_{label}'] = gaussian_component
    fit_csv_path = os.path.join(folder_path, f"{filename}_fitted_values.csv")
    fit_df.to_csv(fit_csv_path, index=False)
    experiment_numbers = df.columns[1:]
    experiment_time = np.arange(len(experiment_numbers)) * 1.1

    integrated_areas = {f'Peak {i}': [] for i in range(num_peaks)}
    
    reciprocal_cm = df.iloc[:, 0]
    
    for experiment_number in experiment_numbers:
        
        start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
        end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]
    
        trimmed_wavenumbers = df.iloc[start_index:end_index+1, 0].values
        spectrum = df.iloc[start_index:end_index+1, experiment_number].values
        
        index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
        index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
        
        slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
        intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
        background_array = slope * trimmed_wavenumbers + intercept
        spectrum_corrected = spectrum - background_array
        popt, _ = curve_fit(combined_function, trimmed_wavenumbers, spectrum_corrected, p0=initial_params, maxfev=10000)
        amps = popt
        gaussians = [gaussian(trimmed_wavenumbers, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values)]

        for i in range(num_peaks):
            area = trapz(gaussians[i], trimmed_wavenumbers)
            integrated_areas[f'Peak {i}'].append(area)
                
    #%%  Saving instegrated areas as csv
    
    integrated_areas_df = pd.DataFrame(integrated_areas)
    integrated_areas_df['Time (s)'] = experiment_time
    header_mapping = {f'Peak {i}': f'Mean {mean_values[i]}' for i in range(num_peaks)}
    integrated_areas_df.rename(columns=header_mapping, inplace=True)
    integrated_areas_df = integrated_areas_df[['Time (s)'] + [col for col in integrated_areas_df.columns if col != 'Time (s)']]
    csv_filename = os.path.join(folder_path, f"{filename}_integrated_areas.csv")
    integrated_areas_df.to_csv(csv_filename, index=False)
    fig, ax = plt.subplots(figsize=(18, 8))
    
    for i, peak_label in enumerate(integrated_areas.keys()):
        plt.plot(experiment_time, integrated_areas[peak_label], label=mean_values[i], color=colors[i])
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18, ncol=3)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Integrated Area (a.u.)', fontsize=18)
    #plt.ylim(-6,5)
    #plt.xlim(0,1000)
    intersections = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)
    

    
    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]
    else: # if error
        text_labels = ["-0.4 V", "-0.8 V"]
    
    
    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2]  # Alternate between the two text labels
    
    
        text_x = (x_start + x_end) / 2  # Calculate the x-coordinate for the text label
        plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center', fontsize=16)
    
    folder = os.path.dirname(file_path)
    folder_path = os.path.join(folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_integration of deconvoluted peak"
    
    png_path = os.path.join(folder_path, f"{filename}.png")
    eps_path = os.path.join(folder_path, f"{filename}.eps")
    svg_path = os.path.join(folder_path, f"{filename}.svg")
    
    plt.subplots_adjust(top=0.9)  # You can modify this value to suit your needs
       
    plt.savefig(png_path)
    plt.savefig(eps_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    
    html_path = os.path.join(folder_path, f"{filename}.html")
    html_content = mpld3.fig_to_html(plt.gcf())  # Convert Matplotlib figure to HTML
    with open(html_path, "w") as html_file:
        html_file.write(html_content)

    
    plt.show()
    
    folder = os.path.dirname(file_path)
    folder_path = os.path.join(folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_integrated areas"
    
    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")
    
    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    
    html_path = os.path.join(folder_path, f"{filename}.html")
    html_content = mpld3.fig_to_html(plt.gcf())  # Convert Matplotlib figure to HTML
    with open(html_path, "w") as html_file:
        html_file.write(html_content)
        
    plt.show()
    

if __name__ == "__main__":
    # Subfolders to include
    include_folders = include_folders

    # Base directory
    base_dir = base_dir

    for folder_name in include_folders:
        # Path to the Reconstruction_based_on_CO_peak_in_eigenspectra folder
        reconstruction_path = os.path.join(base_dir, folder_name, "Reconstruction_based_on_CO_peak_in_eigenspectra")
        
        if not os.path.exists(reconstruction_path):
            print(f"Folder not found: {reconstruction_path}")
            continue

        # Gather all .csv files that start with "DS" or "Recon"
        csv_files = []
        for root, _, files in os.walk(reconstruction_path):
            csv_files.extend([
                os.path.join(root, file) 
                for file in files 
                if file.endswith(".csv") and (file.startswith("DS") or file.startswith("CZ") or file.startswith("ReconstructedData"))
            ])

        if not csv_files:
            print(f"No matching CSV files found in {reconstruction_path}")
            continue

        for csv_file in csv_files:
            print(f"Processing file: {csv_file}")
            experiment_classification = '_unknown'
            for suffix in ["_07", "_08", "_09"]:
                if f"{folder_name}{suffix}.csv" in csv_file:
                    experiment_classification = suffix
                    break

            process_spectrum_file(
                file_path=csv_file,
                experiment_classification=experiment_classification,
                **conditions
            )
if r2_results:
    r2_df = pd.DataFrame(r2_results)
    r2_csv_path = os.path.join(base_dir, "r2_values.csv")
    r2_df.to_csv(r2_csv_path, index=False)
    print(f"R^2 values saved to: {r2_csv_path}")
else:
    print("No R^2 values to save.")

                
#%% now extract the integrated areas at time 195.8

import os
import pandas as pd
import numpy as np
results = []
processed_files = set()  # Set to track processed file paths

# Specify the target peaks and time
target_peaks = [3680, 3520, 3360, 3210, 3100, 2870]
target_time = 195.8
def find_integrated_area_files(base_path, ignore_folders):
    integrated_area_files = []
    for root, dirs, files in os.walk(base_path):  # Recursively walk through folders
        # Exclude ignored folder names
        dirs[:] = [d for d in dirs if d.lower() not in ignore_folders]
        for file in files:
            if file.endswith("_integrated_areas.csv"):
                integrated_area_files.append(os.path.join(root, file))
    return integrated_area_files

ignore_folders = {"raw data", 
                  "1635_peak", 
                  "CO peak", 
                  "900_to_3999", 
                  "650_to_4000", 
                  "2000_to_3999", 
                  #"1101_to_3999", 
                  "non-mean-centered", 
                  "Stark shift"}
for folder_name in include_folders:
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    integrated_files = find_integrated_area_files(folder_path, ignore_folders)
    if not integrated_files:
        print(f"No integrated areas files found in {folder_path}")
        continue

    for full_path in integrated_files:
        file_key = (os.path.basename(full_path), os.path.dirname(full_path))  # Unique identifier: (File, Folder)
        if file_key in processed_files:  # Skip if already processed
            print(f"Skipping duplicate file: {full_path}")
            continue
        processed_files.add(file_key)  # Mark as processed

        print(f"Processing file: {full_path}")
        df_integrated = pd.read_csv(full_path)

        if 'Time (s)' not in df_integrated.columns:
            print(f"'Time (s)' column not found in {full_path}")
            continue

        row = df_integrated[np.isclose(df_integrated['Time (s)'], target_time, atol=0.1)]
        if not row.empty:
            # Extract the top-level folder name for Experiment
            experiment_group = os.path.relpath(full_path, base_dir).split(os.sep)[0]
            print(f"Extracted Experiment: {experiment_group} for file: {full_path}")  # Debugging line

            row_data = {
                'Experiment': experiment_group,
                'File': os.path.basename(full_path),
                'Folder': os.path.dirname(full_path),
            }
            for peak in target_peaks:
                peak_col = f'Mean {peak}'
                row_data[peak_col] = row[peak_col].values[0] if peak_col in row else None
            results.append(row_data)
        else:
            print(f"Target time {target_time} not found in {full_path}")

if results:
    results_df = pd.DataFrame(results)

    # Drop exact duplicates
    results_df = results_df.drop_duplicates()

    # Save the consolidated CSV
    consolidated_csv_path = os.path.join(base_dir, "consolidated_integrated_areas.csv")
    results_df.to_csv(consolidated_csv_path, index=False)
    print(f"Consolidated results saved to: {consolidated_csv_path}")
else:
    print("No data to consolidate. Results list is empty.")
    
    
    
