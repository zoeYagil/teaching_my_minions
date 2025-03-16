import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import seaborn as sns
import os
import matplotlib as mpl
mpl.use('SVG')


mpl.rcParams['svg.fonttype'] = 'none' 

sns.set(style="white")

file_path = '/Users/danielsinausia/Documents/Paper rebuttal/violin like plots discontinuities.csv'
df = pd.read_csv(file_path, header=None)

x_labels = df.iloc[0, 8:13]  
y_values = df.iloc[3, 8:13].astype(float)  
y_errors = df.iloc[4, 8:13].astype(float)  
x_errors = df.iloc[5, 8:13].astype(float)  

x_positions = [0, 5, 10, 15, 20]

fig, ax = plt.subplots(figsize=(8, 6))

def create_violin_polygon(x, y, x_err, y_err):
    y_top = y + y_err
    y_bottom = y - y_err
    x_left = x - x_err
    x_right = x + x_err

    x_top = np.linspace(x_left, x_right, 100)
    y_top_curve = y_top * np.ones_like(x_top)

    x_bottom = np.linspace(x_right, x_left, 100)
    y_bottom_curve = y_bottom * np.ones_like(x_bottom)

    x_violin = np.concatenate([x_top, x_bottom])
    y_violin = np.concatenate([y_top_curve, y_bottom_curve])

    return Polygon(np.column_stack([x_violin, y_violin]), closed=True, color='royalblue', alpha=0.4)

for x, y, x_err, y_err in zip(x_positions, y_values, x_errors, y_errors):
    violin_polygon = create_violin_polygon(x, y, x_err, y_err)
    ax.add_patch(violin_polygon)

    ax.plot(x, y, 'o', color='navy', markersize=8)

ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')

ax.set_aspect('equal', adjustable='datalim')

ax.set_xlabel('X Axis Labels', fontsize=14, fontweight='bold')
ax.set_ylabel('Value', fontsize=14, fontweight='bold')
ax.set_title('Custom Violin-Like Plot', fontsize=16, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.7)

sns.despine()

plt.tight_layout()
plt.show()

output_path = os.path.join(os.path.dirname(file_path), 'violin_like_plot.svg')
plt.savefig(output_path, format='svg')
