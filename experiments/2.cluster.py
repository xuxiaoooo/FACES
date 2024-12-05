import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data[['Depression', 'Anxiety', 'Stress']])

def perform_clustering(data_scaled, n_clusters):
    model = AgglomerativeClustering(n_clusters)
    return model.fit_predict(data_scaled)

def evaluate_clustering(data_scaled, labels):
    silhouette_avg = silhouette_score(data_scaled, labels)
    ch_score = calinski_harabasz_score(data_scaled, labels)
    db_score = davies_bouldin_score(data_scaled, labels)
    return silhouette_avg, ch_score, db_score

def visualize_clusters(data_scaled, labels, n_clusters, output_dir):
    # 为每个簇分配颜色
    color_map = {0: '#8DA0CB', 1: '#F58D63', 2: '#66C2A5'}

    # 生成3D散点图
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_clusters):
        ax.scatter(data_scaled[labels == i, 0],  # X坐标
                   data_scaled[labels == i, 1],  # Y坐标
                   data_scaled[labels == i, 2],  # Z坐标
                   c=color_map[i], label=f'Cluster {i}')
    ax.set_xlabel("Depression", fontweight='bold', fontname='Arial', size=20)
    ax.set_ylabel("Anxiety", fontweight='bold', fontname='Arial', size=20)
    ax.set_zlabel("Stress", fontweight='bold', fontname='Arial', size=20)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tight_layout()

    scatter_plot_path = f'{output_dir}/cluster_{n_clusters}.tif'
    plt.savefig(scatter_plot_path, transparent=True, dpi=600)
    plt.close()

    # 生成饼图
    cluster_counts = np.bincount(labels)
    plt.figure(figsize=(6, 6))
    plt.pie(cluster_counts, 
            colors=[color_map[i] for i in range(n_clusters)], startangle=90)
    plt.tight_layout()

    pie_chart_path = f'{output_dir}/piechart_{n_clusters}.tif'
    plt.savefig(pie_chart_path, transparent=True, dpi=600)
    plt.close()

    return scatter_plot_path, pie_chart_path

def save_cluster_results(df, labels, n_clusters):
    df[f'Cluster_{n_clusters}'] = labels
    return df

# Main code
df = pd.read_csv('../data/baseinfo.csv')
df = df[['cust_id', 'Depression', 'Anxiety', 'Stress']].dropna()

data_scaled = preprocess_data(df)

results = {}
for n_clusters in [2, 3]:
    labels = perform_clustering(data_scaled, n_clusters)
    silhouette_avg, ch_score, db_score = evaluate_clustering(data_scaled, labels)
    scatter_plot_path, pie_chart_path = visualize_clusters(data_scaled, labels, n_clusters, '../draw')
    df = save_cluster_results(df, labels, n_clusters)
    results[f'{n_clusters}_clusters'] = {
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': ch_score,
        'davies_bouldin_score': db_score
    }

# Save final DataFrame to CSV
df_final = df[['cust_id', 'Cluster_2', 'Cluster_3']]
df_final.to_csv('../data/cluster_label.csv', index=False)

print(results)
