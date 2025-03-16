
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit
import os
#import matplotlib
#matplotlib.use('module://matplotlib_inline.backend_inline')

import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'

# =============================================================================
#
# Define a function to apply Savitzky-Golay smoothing
#def apply_savgol_smoothing(df):
#    return df.apply(lambda column: savgol_filter(column, window_length=20, polyorder=2), axis=0)
#
#dfs = {}
#for key, file_path in file_paths.items():
#    dfs[key] = apply_savgol_smoothing(pd.read_csv(file_path, header=None, usecols=range(910)))
#
## Deconvolution of the water stretching signal
#def gaussian(x, amp, mean, sigma):
#    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2)) # This is the mathematical formula for a Gaussian function
#
# =============================================================================
#######################################################################
#                 LOAD THE DATA AND CHOOSE BUBBLED GAS
#######################################################################

# File paths for the CSV files
file_paths = {
    'df': r"/Users/danielsinausia/Documents/Experiments/CZ_00002/Stark shift/ReconstructedData_PCs2&3&4&5&6&7&8&9&10&11&12&13&14&15.csv"
}







#######################################################################
#                 NOT TO CHANGE BELOW THIS POINT
#######################################################################























#%%
folder_path = os.path.dirname(file_paths['df'])
folder_name = os.path.basename(os.path.dirname(file_paths['df']))
files_in_folder = os.listdir(folder_path)
csv_files = [file for file in files_in_folder if file.startswith("DS_") or file.startswith("CZ_") and file.endswith(".csv")]
experiment_classification = '_08'
    
#%%
dfs = {}

for key, file_path in file_paths.items():
    dfs[key] = pd.read_csv(file_path, header=None, skiprows=1)
    dfs[key] = dfs[key].iloc[:, :911]
#    dfs[key]=dfs[key][::-1] #activate for original

#######################################################################
#                        CREATE THE FUNCTIONS
#######################################################################

initial_params = [1, 2078, 1, 2050] #Peak positions obtained from 10.1021/jacsau.1c00281. The choice of 1 and 10 for amplitude and std dev are arbitrary, chosen with the help of chatGPT, but since I am interested in the evolution of the peaks rather than the absolute numbers, I'll go with it
sigma_fixed_1 = 5
sigma_fixed_2 = 5


def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))  # Gaussian function

def combined_function(x, amp1, mean1, amp2, mean2):
    return (
        gaussian(x, amp1, mean1, sigma_fixed_1) +
        gaussian(x, amp2, mean2, sigma_fixed_2)
    )



        
        
 #######################################################################
#               PLOT SINGLE FTIR DECONVOLUTED SPECTRUM
#######################################################################

start_reciprocal_cm = 1900
end_reciprocal_cm = 2200
start_reciprocal_cm_bkg = 2000  
end_reciprocal_cm_bkg = 2200
spectrum_to_plot_as_example = 600


reciprocal_cm = dfs[key].iloc[:, 0]


for key in dfs.keys():
    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]


    experiment_number = spectrum_to_plot_as_example
    trimmed_wavenumbers = dfs[key].iloc[start_index:end_index+1, 0].values
    spectrum = dfs[key].iloc[start_index:end_index+1, experiment_number].values
    
    index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
    index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
    slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
    intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
    background_array = slope * trimmed_wavenumbers + intercept
    spectrum_corrected = spectrum - background_array
    popt, _ = curve_fit(combined_function, reciprocal_cm[start_index:end_index+1], spectrum_corrected, p0=initial_params, maxfev=1000000)
    amp1, mean1, amp2, mean2 = popt
    gaussian_1 = gaussian(reciprocal_cm[start_index:end_index + 1], amp1, mean1, sigma_fixed_1)
    gaussian_2 = gaussian(reciprocal_cm[start_index:end_index + 1], amp2, mean2, sigma_fixed_2)
    fitted_curve = combined_function(reciprocal_cm[start_index:end_index+1], *popt)
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.plot(reciprocal_cm[start_index:end_index + 1], gaussian_1, color='#40506c')
    plt.fill_between(reciprocal_cm[start_index:end_index + 1], gaussian_1, color='#40506c', alpha=0.6)
    plt.plot(reciprocal_cm[start_index:end_index + 1], gaussian_2, color='#708090')
    plt.fill_between(reciprocal_cm[start_index:end_index + 1], gaussian_2, color='#708090', alpha=0.6)

    plt.plot(reciprocal_cm[start_index:end_index + 1], spectrum_corrected, label='Original Data')
    plt.plot(reciprocal_cm[start_index:end_index + 1], fitted_curve, label='Fitted Curve')
    plt.xlabel('Wavenumbers (cm$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticks([])
    plt.tight_layout()
    plt.title(f'Spectrum {experiment_number}')
    plt.xlim(end_reciprocal_cm, start_reciprocal_cm)

    folder = os.path.dirname(file_paths[key])
    folder_path = os.path.join(folder, 'CO_peak')
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder)

    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")

    
    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    plt.show()


#######################################################################
#               EVOLUTION OF THE INTEGRATED AREAS
#######################################################################

experiment_numbers = dfs[key].columns[1:]
experiment_time = np.arange(len(experiment_numbers)) * 1.1

integrated_areas_1 = {}
integrated_areas_2 = {}


for key in dfs.keys():
    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]
    integrated_areas_1[key] = []
    integrated_areas_2[key] = []
    peak_positions_1 = []
    peak_positions_2 = []
    for experiment_number in experiment_numbers:
        trimmed_wavenumbers = dfs[key].iloc[start_index:end_index+1, 0].values
        spectrum = dfs[key].iloc[start_index:end_index+1, experiment_number].values
        index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
        index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
        slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
        intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
        background_array = slope * trimmed_wavenumbers + intercept
        spectrum_corrected = spectrum - background_array
        popt, _ = curve_fit(combined_function, reciprocal_cm[start_index:end_index+1], spectrum_corrected, p0=initial_params, maxfev=1000000)
        amp1, mean1, amp2, mean2 = popt
        gaussian_1 = gaussian(reciprocal_cm[start_index:end_index + 1], amp1, mean1, sigma_fixed_1)
        gaussian_2 = gaussian(reciprocal_cm[start_index:end_index + 1], amp2, mean2, sigma_fixed_2)
        area_1 = trapz(gaussian_1, reciprocal_cm[start_index:end_index + 1])
        area_2 = trapz(gaussian_2, reciprocal_cm[start_index:end_index + 1])
        integrated_areas_1[key].append(area_1)
        integrated_areas_2[key].append(area_2)
        peak_positions_1.append(mean1)
        peak_positions_2.append(mean2)


        


    integrated_areas_df = pd.DataFrame({'Area1': integrated_areas_1[key], 'Area2': integrated_areas_2[key]})
    integrated_areas_df['Time (s)'] = experiment_time
    integrated_areas_df = integrated_areas_df[['Time (s)'] + [col for col in integrated_areas_df.columns if col != 'Time (s)']]
    csv_filename = os.path.join(folder_path, f"{filename}_integrated_areas.csv")
    integrated_areas_df.to_csv(csv_filename, index=False)

    fig, ax = plt.subplots(figsize=(18, 8))
    plt.plot(experiment_time, integrated_areas_1[key], label='Area 1', color='#40506c')
    plt.plot(experiment_time, integrated_areas_2[key], label='Area 2', color='#708090')

    plt.xlabel('Time (s)')
    plt.ylabel('Integrated Area (a.u.)')
    plt.xlim(0, experiment_time[-1])


# =============================================================================
#     plt.gca().yaxis.set_ticklabels([])
#     plt.gca().yaxis.set_ticks([])
# =============================================================================
    plt.legend()
    plt.tight_layout()
# =============================================================================
#     plt.title('Integrated Areas of Hidden Peaks')
#     
# =============================================================================
#######################################################################
#                               PULSES
#######################################################################
    
    intersections = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)
    
    
    
    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]
    
    
    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2]  # Alternate between the two text labels
    
    
        text_x = (x_start + x_end) / 2  # Calculate the x-coordinate for the text label
        plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center')
        
        
    folder = os.path.dirname(file_paths[key])
    folder_path = os.path.join(folder, 'CO_peak')
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_integration"
    
    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")
    
    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    plt.show()


#######################################################################
#               EVOLUTION OF THE PEAK POSITIONS
#######################################################################

    
    
    peak_positions_1 = np.array(peak_positions_1)
    peak_positions_2 = np.array(peak_positions_2)

    peak_positions_df = pd.DataFrame({'Peak1': peak_positions_1, 'Peak2': peak_positions_2})
    peak_positions_df['Time (s)'] = experiment_time
    peak_positions_df = peak_positions_df[['Time (s)'] + [col for col in peak_positions_df.columns if col != 'Time (s)']]
    peak_positions_csv_filename = os.path.join(folder_path, f"{filename}_peak_shift.csv")
    peak_positions_df.to_csv(peak_positions_csv_filename, index=False)

    fig, ax = plt.subplots(figsize=(18, 8))
    plt.plot(experiment_time, peak_positions_1, label='Peak 1', color='#40506c')
    plt.plot(experiment_time, peak_positions_2, label='Peak 2', color='#708090')
    plt.xlabel('Time (s)')
    plt.ylabel('Wavenumbers (cm$^{-1}$)')
    plt.legend()
    plt.xlim(0, experiment_time[-1])


    #plt.title('Peak Positions vs. Time')
    
    intersections = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)
    
    
    
    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]
    
    
    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2]  # Alternate between the two text labels
    
    
        text_x = (x_start + x_end) / 2  # Calculate the x-coordinate for the text label
        plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center')


    folder = os.path.dirname(file_paths[key])
    folder_path = os.path.join(folder, 'CO_peak')
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_peak_shift"
    
    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")
    
    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    plt.show()
