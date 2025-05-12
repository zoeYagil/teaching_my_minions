'''
this is clustering all the experiments together.

IF THERE IS AN ERROR: check the following after running it once: that the number of clusters in cluster_to_recluster (in the code twice)
and n_clusters match the value shown in the terminal in "Optimal Number of Clusters:"

'''



import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths




input_file = '/Users/danielsinausia/Documents/Experiments/DTW/DTW_Analysis_SG_Smoothed_new_curves_normalizing_every_curve/PC2-15_first_pulse_short.csv'
data = pd.read_csv(input_file)


experiments = data.columns[2:]
combined_data = []
for experiment in experiments:
    alignment_data = data[[data.columns[0], data.columns[1], experiment]]
    pivot_data = alignment_data.pivot(index='PC', columns='Reference', values=experiment)
    pivot_data = pivot_data.fillna(0)  # Replace NaN with 0
    pivot_data.index = [f"{pc}_{experiment}" for pc in pivot_data.index]
    combined_data.append(pivot_data)
combined_data = pd.concat(combined_data, axis=0)
combined_data = combined_data.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric
combined_data = combined_data.dropna()  # Drop rows with non-numeric values
combined_data.columns = combined_data.columns.astype(str)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(combined_data)

similarity_matrix = rbf_kernel(normalized_data)
n_clusters = 2  # Adjust number of clusters as needed
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
labels = clustering.fit_predict(similarity_matrix)

combined_data['Cluster'] = labels



output_dir = os.path.dirname(input_file)
output_file = os.path.join(output_dir, 'combined_clustering_results.csv')
combined_data.to_csv(output_file)

print(f"Clustering results saved to {output_file}")

fig, ax = plt.subplots(figsize=(18, 8))
scatter = ax.scatter(
    range(len(labels)), 
    labels, 
    c=labels, 
    cmap='viridis', 
    alpha=0.8
)
ax.set_xlabel('Time-series (PCs)', fontsize=18)
ax.set_ylabel('Cluster Label', fontsize=18)
ax.set_title('Clustering of PCs Across All Experiments', fontsize=18)
plt.colorbar(scatter, ax=ax, label='Cluster Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clustering_visualization.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'clustering_visualization.svg'))
plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = {}
for n_clusters in range(2, 11):  # Test from 2 to 10 clusters
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    labels = clustering.fit_predict(similarity_matrix)
    score = silhouette_score(normalized_data, labels, metric='euclidean')  # Use Euclidean distance
    silhouette_scores[n_clusters] = score
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
print(f"Optimal Number of Clusters: {optimal_clusters}")
clustering = SpectralClustering(n_clusters=optimal_clusters, affinity='precomputed', random_state=0)
labels = clustering.fit_predict(similarity_matrix)
combined_data['Cluster'] = labels

output_dir = os.path.dirname(input_file)
output_file = os.path.join(output_dir, 'combined_clustering_results.csv')
combined_data.to_csv(output_file)

print(f"Clustering results saved to {output_file}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(
    list(silhouette_scores.keys()), 
    list(silhouette_scores.values()), 
    marker='o', 
    linestyle='-', 
    linewidth=2
)
ax.set_xlabel('Number of Clusters', fontsize=18)
ax.set_ylabel('Silhouette Score', fontsize=18)
ax.set_title('Silhouette Scores for Different Numbers of Clusters', fontsize=18)
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'silhouette_scores.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'silhouette_scores.svg'))

plt.show()

from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(normalized_data, labels)
print(f"Davies-Bouldin Index: {db_index}")

centroids = combined_data.groupby('Cluster').mean()
centroids_output_file = os.path.join(output_dir, 'cluster_centroids.csv')
centroids.to_csv(centroids_output_file)
print(f"Cluster centroids saved to {centroids_output_file}")
fig, ax = plt.subplots(figsize=(18, 8))
for cluster in centroids.index:
    ax.plot(
        centroids.columns,  # Feature names
        centroids.loc[cluster],  # Data for this cluster
        label=f'Cluster {cluster}', 
        marker='o', 
        linewidth=2
    )
ax.set_xlabel('Features (Reference Curves)', fontsize=18)
ax.set_ylabel('Mean Alignment Costs', fontsize=18)
ax.set_title('Cluster Centroids', fontsize=18)
ax.legend(title='Clusters', fontsize=14, loc='upper right')
ax.grid(True)
plt.tight_layout()

centroid_plot_file_base = os.path.join(output_dir, 'cluster_centroids')
plt.savefig(f"{centroid_plot_file_base}.png", dpi=300)
plt.savefig(f"{centroid_plot_file_base}.svg")
plt.show()

centroid_plot_file = os.path.join(output_dir, 'cluster_centroids.png')
plt.savefig(centroid_plot_file, dpi=300)
print(f"Centroid plot saved to {centroid_plot_file}")

plt.show()

#%% cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

centroids = combined_data.groupby('Cluster').mean()
if 'Cluster' in centroids.columns:
    centroids = centroids.drop(columns=['Cluster'])

similarity_scores = cosine_similarity(normalized_data, centroids)
similarity_df = pd.DataFrame(
    similarity_scores, 
    columns=[f"Cluster_{int(cluster)}" for cluster in centroids.index],
    index=combined_data.index
)

similarity_df['Assigned_Cluster'] = combined_data['Cluster']

similarity_output_file = os.path.join(output_dir, 'time_series_cluster_similarity.csv')
similarity_df.to_csv(similarity_output_file)
print(f"Time-series cluster similarity saved to {similarity_output_file}")

#%% regression

from sklearn.linear_model import LinearRegression
import numpy as np

regression_scores = []
for i, row in enumerate(normalized_data):
    scores = []
    for centroid in centroids.values:
        model = LinearRegression()
        model.fit(centroid.reshape(-1, 1), row)
        r2_score = model.score(centroid.reshape(-1, 1), row)
        scores.append(r2_score)
    regression_scores.append(scores)
regression_df = pd.DataFrame(
    regression_scores, 
    columns=[f"Cluster_{int(cluster)}" for cluster in centroids.index],
    index=combined_data.index
)

regression_df['Assigned_Cluster'] = combined_data['Cluster']
regression_output_file = os.path.join(output_dir, 'time_series_cluster_regression.csv')
regression_df.to_csv(regression_output_file)
print(f"Time-series regression scores saved to {regression_output_file}")

#%%

import seaborn as sns

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

normalized_data = combined_data.copy()
for column in combined_data.columns[:-1]:  # Skip the 'Cluster' column
    normalized_data[column] = (combined_data[column] - combined_data[column].min()) / (
        combined_data[column].max() - combined_data[column].min()
    )

sorted_data = normalized_data.sort_values('Cluster')
heatmap_data = sorted_data.drop(columns=['Cluster']).T

unique_clusters = sorted_data['Cluster'].unique()
cluster_sizes = sorted_data['Cluster'].value_counts(sort=False)
cluster_boundaries = np.cumsum(cluster_sizes)

fig, ax = plt.subplots(figsize=(100, 10))  # Increase the figure size for better label visibility
sns.heatmap(
    heatmap_data,            # Transposed data for plotting
    cmap='Spectral_r',       # Colormap
    cbar_kws={'label': 'Normalized Values'},  # Label for the colorbar
    annot=False,             # Turn on/off annotations
    linewidths=2,          # Add cell borders for better distinction
    linecolor='gray'
)

for boundary in cluster_boundaries[:-1]:  # Skip the last boundary
    ax.axvline(boundary, color='black', linewidth=1.5, linestyle='--')

ax.set_xlabel('Experiments', fontsize=18)
ax.set_ylabel('Features (Curves)', fontsize=18)
ax.set_title('Normalized Heatmap with Cluster Separators', fontsize=18)

ax.set_xticks(range(len(heatmap_data.columns)))  # Ensure all x-axis ticks are shown
ax.set_xticklabels(heatmap_data.columns, fontsize=10, rotation=45, ha='right')  # Rotate labels for better fit
ax.set_yticklabels(heatmap_data.index, fontsize=14)

plt.tight_layout()

heatmap_output_base = os.path.join(output_dir, 'normalized_heatmap_by_cluster')
plt.savefig(f"{heatmap_output_base}.png", dpi=300)
plt.savefig(f"{heatmap_output_base}.svg")
plt.show()


#%%

def recluster_data(data, cluster_to_recluster, output_dir):
    # Check if the cluster exists
    if cluster_to_recluster not in data['Cluster'].unique():
        print(f"Cluster {cluster_to_recluster} does not exist. Skipping reclustering.")
        return None

    subset = data[data['Cluster'] == cluster_to_recluster].drop(columns=['Cluster'])
    subset.columns = subset.columns.astype(str)
    recluster_dir = os.path.join(output_dir, f"recluster_cluster_{cluster_to_recluster}")
    os.makedirs(recluster_dir, exist_ok=True)
    scaler = StandardScaler()
    normalized_subset = scaler.fit_transform(subset)
    similarity_matrix = rbf_kernel(normalized_subset)
    n_reclusters = 2  # Adjust the number of clusters for reclustering
    reclustering = SpectralClustering(n_clusters=n_reclusters, affinity='precomputed', random_state=0)
    recluster_labels = reclustering.fit_predict(similarity_matrix)
    subset['Recluster'] = recluster_labels
    recluster_file = os.path.join(recluster_dir, f"recluster_results.csv")
    subset.to_csv(recluster_file)
    print(f"Reclustering results saved to {recluster_file}")
    
    silhouette_scores = {}
    for n_clusters in range(2, 11):
        temp_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
        temp_labels = temp_clustering.fit_predict(similarity_matrix)
        score = silhouette_score(normalized_subset, temp_labels, metric='euclidean')
        silhouette_scores[n_clusters] = score
        print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
    
    optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Optimal Number of Clusters for Reclustering: {optimal_clusters}")
    
    # Davies-Bouldin Index
    db_index = davies_bouldin_score(normalized_subset, recluster_labels)
    print(f"Davies-Bouldin Index for Reclustering: {db_index}")
    
    silhouette_file = os.path.join(recluster_dir, 'silhouette_scores.csv')
    pd.DataFrame.from_dict(silhouette_scores, orient='index', columns=['Silhouette Score']).to_csv(silhouette_file)
    print(f"Silhouette scores saved to {silhouette_file}")
    
    centroids = subset.groupby('Recluster').mean()
    centroids_output_file = os.path.join(recluster_dir, 'cluster_centroids.csv')
    centroids.to_csv(centroids_output_file)
    print(f"Recluster centroids saved to {centroids_output_file}")
    
    fig, ax = plt.subplots(figsize=(18, 8))
    for cluster in centroids.index:
        ax.plot(
            centroids.columns, 
            centroids.loc[cluster], 
            label=f'Recluster {cluster}', 
            marker='o', 
            linewidth=2
        )
    ax.set_xlabel('Features (Reference Curves)', fontsize=18)
    ax.set_ylabel('Mean Alignment Costs', fontsize=18)
    ax.set_title(f'Recluster Centroids for Cluster {cluster_to_recluster}', fontsize=18)
    ax.legend(title='Clusters', fontsize=14, loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    centroid_plot_file = os.path.join(recluster_dir, 'recluster_centroids.png')
    plt.savefig(centroid_plot_file, dpi=300)
    plt.savefig(centroid_plot_file.replace('.png', '.svg'))
    plt.show()
    
    heatmap_data = subset.sort_values('Recluster').drop(columns=['Recluster']).T
    unique_clusters = subset['Recluster'].unique()
    cluster_sizes = subset['Recluster'].value_counts(sort=False)
    cluster_boundaries = np.cumsum(cluster_sizes)
    
    fig, ax = plt.subplots(figsize=(100, 10))
    sns.heatmap(
        heatmap_data,
        cmap='Spectral_r',
        cbar_kws={'label': 'Normalized Values'},
        annot=False,
        linewidths=2,
        linecolor='gray'
    )
    for boundary in cluster_boundaries[:-1]:
        ax.axvline(boundary, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Subset Experiments', fontsize=18)
    ax.set_ylabel('Features (Curves)', fontsize=18)
    ax.set_title(f'Recluster Heatmap for Cluster {cluster_to_recluster}', fontsize=18)
    plt.tight_layout()
    heatmap_file = os.path.join(recluster_dir, 'recluster_heatmap.png')
    plt.savefig(heatmap_file, dpi=300)
    plt.savefig(heatmap_file.replace('.png', '.svg'))
    plt.show()

    return recluster_dir

recluster_dir = recluster_data(combined_data, cluster_to_recluster=2, output_dir=output_dir)
print(f"Reclustering analysis completed and saved to {recluster_dir}")





