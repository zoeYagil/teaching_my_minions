'''

In the dictionary below "file_to_pH", add the name of your csv file and the pH at which you performed your experiment, as seen in the examples that are there.
In csv_directory, add the directory to the FOLDER where your csv files are


'''




import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
import matplotlib as mpl
mpl.use('SVG')

mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths


csv_directory = r"/Users/danielsinausia/Downloads/Potentiostat 2/"

file_to_pH = {
    'DS_00145_08.csv': 6.86,
    'DS_00184_13.csv': 7.45,
    'DS_00185_13.csv': 6.7,
    'DS_00186_13.csv': 7.18,
    'DS_00187_13.csv': 7.16,
    'DS_00188_13.csv': 7.16,
    'DS_00189_13.csv': 6.56,
    'DS_00190_13.csv': 5.86,
    'DS_00191_13.csv': 7.8,
    'DS_00192_13.csv': 7.8,
    'DS_00197_13.csv': 7.97
}


#%% NOT TO CHANGE ANYTHING BELOW THIS POINT


















output_folder = os.path.join(csv_directory, "Currents plotted")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        csv_file_path = os.path.join(csv_directory, filename)
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file_path, encoding='latin-1')
        
        num_columns = len(df.columns)
        if num_columns != 2:
            print(f"File '{filename}' has {num_columns} columns instead of 2.")
            print(f"Columns: {df.columns}")


diameter = 7  # mm
area = (np.pi * ((diameter/1000) / 2) ** 2)

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        csv_file_path = os.path.join(csv_directory, filename)
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file_path, encoding='latin-1')

        script_file_name = os.path.splitext(filename)[0]
        df = df.iloc[1:, :]
        df.columns = ['Time (s)', 'Current (µA)']
        df = df.astype(float)
        pH = file_to_pH.get(filename, 7.0)
        conversion_factor_RHE = 0.197 + 0.059 * pH

        plt.figure()
        plt.plot(df['Time (s)'], df['Current (µA)'])
        plt.xlabel('Time (s)')
        plt.ylabel('Current (µA)')
        plt.xlim(df['Time (s)'].min(), df['Time (s)'].max())
        plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useMathText=False))
        file_name = os.path.join(output_folder, script_file_name + 'current')
        plt.savefig(file_name + ".png", dpi=300)
        plt.savefig(file_name + ".svg")
        plt.close()
        df['Current Density (A/m^2)'] = (df['Current (µA)'] / 1_000_000) / area
        plt.figure()
        plt.plot(df['Time (s)'], df['Current Density (A/m^2)'])
        plt.xlabel('Time (s)')
        plt.ylabel('j (A/m^2)')
        plt.xlim(df['Time (s)'].min(), df['Time (s)'].max())
        plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useMathText=False))
        plt.ticklabel_format(axis='y', style='plain')
        file_name = os.path.join(output_folder, script_file_name + 'density_current')
        plt.savefig(file_name + ".png", dpi=300)
        plt.savefig(file_name + ".svg")
        plt.close()

# plt.show()
