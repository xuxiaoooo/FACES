import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D

def perform_kmeans(data, n_clusters):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    ).fit(data)
    return kmeans.labels_

def plot_3d(data, labels, filename):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    color_map = {0: '#8DA0CB', 1: '#F58D63', 2: '#66C2A5'}
    colors = [color_map[label] for label in labels]
    ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, alpha=0.6)
    ax.set_xlabel("Depression", fontweight='bold', fontname='Arial', size=20)
    ax.set_ylabel("Anxiety", fontweight='bold', fontname='Arial', size=20)
    ax.set_zlabel("Stress", fontweight='bold', fontname='Arial', size=20)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tight_layout()
    plt.savefig(filename, transparent=True, dpi=600)

def evaluate_clustering(labels, data):
    silhouette_avg = silhouette_score(data, labels)
    db_index = davies_bouldin_score(data, labels)
    ch_index = calinski_harabasz_score(data, labels)
    return silhouette_avg, db_index, ch_index

if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv('../data/baseinfo.csv')
    df = df[['cust_id', 'Depression', 'Anxiety', 'Stress']].dropna()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[['Depression', 'Anxiety', 'Stress']])

    # Clustering
    df['Cluster_2'] = perform_kmeans(data_scaled, 2)
    df['Cluster_3'] = perform_kmeans(data_scaled, 3)
    clusters_df = df[['cust_id', 'Cluster_2', 'Cluster_3']]
    clusters_df.to_csv('../data/clusters.csv', index=False)

    # Save plots
    plot_3d(data_scaled, df['Cluster_2'], '../draw/cluster_2.png')
    plot_3d(data_scaled, df['Cluster_3'], '../draw/cluster_3.png')

    s2, db2, ch2 = evaluate_clustering(df['Cluster_2'], data_scaled)
    print(f"2 Clusters: Silhouette Coefficient: {s2:.3f}, Davies-Bouldin Index: {db2:.3f}, Calinski-Harabasz Index: {ch2:.3f}")

    # Evaluate clustering for 3 clusters
    s3, db3, ch3 = evaluate_clustering(df['Cluster_3'], data_scaled)
    print(f"3 Clusters: Silhouette Coefficient: {s3:.3f}, Davies-Bouldin Index: {db3:.3f}, Calinski-Harabasz Index: {ch3:.3f}")