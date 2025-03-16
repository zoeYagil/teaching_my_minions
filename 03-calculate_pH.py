import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'


file_path = '/Users/danielsinausia/Documents/Experiments/DS_00145/PC2-15/1101_to_3999/PC2-15_withoutbackground_correction_integrated_areas.csv'


############################################################################################################
################## NOT TO CHANGE ANYTHING BELOW THIS LINE######################################################
############################################################################################################






df = pd.read_csv(file_path, skiprows=0)
parent_dir = os.path.dirname(file_path)

mean_1430_col = [col for col in df.columns if "Mean 1430" in col][0]
mean_1368_col = [col for col in df.columns if "Mean 1368" in col][0]

df['Ratio_1430_1368'] = df[mean_1430_col] / df[mean_1368_col]
time = df.iloc[:, 0]

output_csv = os.path.join(parent_dir, 'ratio_1430_1368.csv')
df.to_csv(output_csv, index=False)

fig, ax = plt.subplots()
ax.plot(time, df['Ratio_1430_1368'], label='Ratio 1430/1368')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Ratio 1430/1368')
ax.set_title('Ratio of Mean 1430 to Mean 1368')
ax.legend()
#ax.set_ylim(-20, 20)
#ax.set_xlim(0, 1000)

output_png = os.path.join(parent_dir, 'ratio_1430_1368_plot.png')
output_svg = os.path.join(parent_dir, 'ratio_1430_1368_plot.svg')
fig.savefig(output_png, format='png')
fig.savefig(output_svg, format='svg')

plt.show()
