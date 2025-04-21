'''
you can find where the csv files are saved in prints in the console
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import spearmanr
import seaborn as sns
import os

import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'

#%% THINGS TO CHANGE
cmap = 'magma'

file_configs = [
    {
        'experiment': 'DS_00132',
        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00132/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 15,
        'polyorder': 2,
        'peak_threshold': 0.5,
        'trough_threshold': 0.5,
        'magnitude_threshold': 0.4
    },
    
    {
        'experiment': 'DS_00133',
        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00133/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 5,
        'polyorder': 2,
        'peak_threshold': 2,
        'trough_threshold': 1,
        'magnitude_threshold': 0.8
    },

    {
        'experiment': 'DS_00127',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00127/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 5,
        'polyorder': 2,
        'peak_threshold': 0.3,
        'trough_threshold': 0.3,
        'magnitude_threshold': 1.4
    },
    
    {
        'experiment': 'DS_00163',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00163/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 15,
        'polyorder': 2,
        'peak_threshold': 1,
        'trough_threshold': 1,
        'magnitude_threshold': 0.43
    },

    {
        'experiment': 'DS_00131',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00131/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 15,
        'polyorder': 2,
        'peak_threshold': 1,
        'trough_threshold': 1,
        'magnitude_threshold': 0.6
    },
    
    {
        'experiment': 'DS_00138',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00138/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 9,
        'polyorder': 2,
        'peak_threshold': 0.2,
        'trough_threshold': 0.2,
        'magnitude_threshold': 0.75
    },
    
    {
        'experiment': 'DS_00135',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/currentSpearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00135/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 19,
        'polyorder': 2,
        'peak_threshold': 0.1,
        'trough_threshold': 0.1,
        'magnitude_threshold': 0.26
    },
    
    {
        'experiment': 'DS_00139',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00139/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 7,
        'polyorder': 2,
        'peak_threshold': 0.2,
        'trough_threshold': 0.2,
        'magnitude_threshold': 1
    },
    
    {
        'experiment': 'DS_00136',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00136/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 7,
        'polyorder': 2,
        'peak_threshold': 10,
        'trough_threshold': 10,
        'magnitude_threshold': 1.27
    },
    
    {
        'experiment': 'DS_00140',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00140/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 10,
        'polyorder': 2,
        'peak_threshold': 0.1,
        'trough_threshold': 0.1,
        'magnitude_threshold': 0.92
    },

    {
        'experiment': 'DS_00137',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00137/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 7,
        'polyorder': 2,
        'peak_threshold': 10,
        'trough_threshold': 10,
        'magnitude_threshold': 4.5
    },
    
    {
        'experiment': 'DS_00141',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00141/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 9,
        'polyorder': 2,
        'peak_threshold': 0.5,
        'trough_threshold': 0.5,
        'magnitude_threshold': 0.83
    },

    {
        'experiment': 'DS_00144',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00144/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 10,
        'polyorder': 2,
        'peak_threshold': 0.5,
        'trough_threshold': 0.5,
        'magnitude_threshold': 0.7
    },

    {
        'experiment': 'DS_00142',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00142/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 13,
        'polyorder': 2,
        'peak_threshold': 0.2,
        'trough_threshold': 0.1,
        'magnitude_threshold': 0.23
    },
    
    {
        'experiment': 'DS_00145',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00145/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 5,
        'polyorder': 2,
        'peak_threshold': 15,
        'trough_threshold': 15,
        'magnitude_threshold': 2
    },

    {
        'experiment': 'DS_00143',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00143/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 10,
        'polyorder': 2,
        'peak_threshold': 0.3,
        'trough_threshold': 0.3,
        'magnitude_threshold': 0.5
    },

    {
        'experiment': 'DS_00146',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00146/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 10,
        'polyorder': 2,
        'peak_threshold': 1.4,
        'trough_threshold': 1.4,
        'magnitude_threshold': 1.4
    },

    {
        'experiment': 'DS_00181',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00181/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 15,
        'polyorder': 2,
        'peak_threshold': 1,
        'trough_threshold': 1,
        'magnitude_threshold': 0.3
    },


    {
        'experiment': 'DS_00180',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00180/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 25,
        'polyorder': 2,
        'peak_threshold': 1,
        'trough_threshold': 1,
        'magnitude_threshold': 0.1
    },
    
    {
        'experiment': 'DS_00148',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00148/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 10,
        'polyorder': 2,
        'peak_threshold': 1,
        'trough_threshold': 1,
        'magnitude_threshold': 0.9
    },

    {
        'experiment': 'DS_00152',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00152/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 20,
        'polyorder': 2,
        'peak_threshold': 0.6,
        'trough_threshold': 0.6,
        'magnitude_threshold': 0.22
    },

    {
        'experiment': 'DS_00149',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00149/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 25,
        'polyorder': 2,
        'peak_threshold': 1,
        'trough_threshold': 1,
        'magnitude_threshold': 0.2
    },

    {
        'experiment': 'DS_00153',

        'file_path_areas_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/DTW_Analysis_Report.csv',
        'file_path_areas_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/areas/Spearman_Correlation_Report.csv',
        'file_path_current_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/DTW_Analysis_Report.csv',
        'file_path_current_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/current/Spearman_Correlation_Report.csv',
        'file_path_pH_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/DTW_Analysis_Report.csv',
        'file_path_pH_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/pH/Spearman_Correlation_Report.csv',
        'file_path_stark_dtw': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/DTW_Analysis_Report.csv',
        'file_path_stark_spearman': '/Users/danielsinausia/Documents/Experiments/DS_00153/Reconstruction_based_on_CO_peak_in_eigenspectra/Interfacial_layer/PC1_discontinuities/stark/Spearman_Correlation_Report.csv',
 
        'window_length': 15,
        'polyorder': 2,
        'peak_threshold': 0.6,
        'trough_threshold': 0.6,
        'magnitude_threshold': 0.3
    },

]


def extract_average_row_and_header(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    header = None
    average_row = None
    
    for i, line in enumerate(lines):
        if "Discontinuity Time (s)" in line:
            header = line.strip().split(",")
        if line.strip().startswith("Average"):
            average_row = line.strip().split(",")
            break
    return header, average_row

all_experiments_data = {category: [] for category in ["areas", "current", "pH", "stark"]}
for config in sorted(file_configs, key=lambda x: x['experiment']):  # Sort experiments alphabetically
    experiment = config['experiment']
    output_folder = os.path.join(os.path.dirname(config['file_path_areas_dtw']), "Averages_Combined")
    os.makedirs(output_folder, exist_ok=True)
    file_pairs = {
        "areas": ('file_path_areas_dtw', 'file_path_areas_spearman'),
        "current": ('file_path_current_dtw', 'file_path_current_spearman'),
        "pH": ('file_path_pH_dtw', 'file_path_pH_spearman'),
        "stark": ('file_path_stark_dtw', 'file_path_stark_spearman')
    }
    for category, (dtw_key, spearman_key) in file_pairs.items():
        combined_rows = []
        print(f"\n Processing {experiment} - {category}")
        
        for file_key in [dtw_key, spearman_key]:
            file_path = config.get(file_key)  
            if file_path and os.path.exists(file_path):  
                print(f" Found file: {file_path}")
                header, average_row = extract_average_row_and_header(file_path)
                if average_row:
                    print(f"'Average' row found in {file_key}, adding to dataset.")
                    df = pd.DataFrame([average_row])  # Convert list to DataFrame
                    df.insert(0, 'Source', os.path.basename(file_path))  # Add filename as source
                    df.insert(0, 'Experiment', experiment)  # Add experiment as first column
                    combined_rows.append(df)
                else:
                    print(f"No 'Average' row found in: {file_path}")
        
        if combined_rows:
            final_df = pd.concat(combined_rows, ignore_index=True)
            if header:
                column_names = ["Experiment", "Source"] + header 
            else:
                column_names = ["Experiment", "Source"] + [f"Column_{i}" for i in range(final_df.shape[1] - 2)]
            final_df.columns = column_names
            all_experiments_data[category].append(final_df)
            
            output_file = os.path.join(output_folder, f"{experiment}_{category}_Averages.csv")
            final_df.to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file}")
        else:
            print(f"⚠️ No valid data found for {category} in experiment {experiment}. Skipping save.")

for category, dfs in all_experiments_data.items():
    if dfs:
        combined_experiments_df = pd.concat(dfs, ignore_index=True)
        combined_output_file = os.path.join(os.path.dirname(file_configs[0]['file_path_areas_dtw']), f"All_Experiments_{category}_Averages.csv")
        combined_experiments_df.to_csv(combined_output_file, index=False)
        print(f"\n All experiments concatenated for {category} and saved to {combined_output_file}")
    else:
        print(f" No data found for {category}. No combined file created.")

print("\n Processing complete. Files saved in their respective 'Averages_Combined' folders and overall combined files per category.")
