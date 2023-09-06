import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_correlation(dataframe):
    correlation_matrix = dataframe.corr(method='pearson')
    print("\nCorrelation Analysis:")
    print(correlation_matrix)
    evaluate_correlation(correlation_matrix)  # 假设这是您已经定义的另一个函数

    # 设置全局字体大小和字体为 Arial
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'

    fig, ax = plt.subplots(figsize=(8, 6))  # 可以调整 figsize 以适应更大的字体
    cax = ax.matshow(correlation_matrix, cmap='BuPu')
    plt.colorbar(cax)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)  # 旋转 x 轴标签以便更好地显示
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    
    for (i, j), z in np.ndenumerate(correlation_matrix):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=22, fontname='Arial',fontweight='bold')
    
    plt.savefig('../draw/correlation.png', dpi=600, bbox_inches='tight', transparent=True)

def evaluate_correlation(correlation_matrix):
    for column in correlation_matrix.columns:
        for index in correlation_matrix.index:
            value = correlation_matrix.loc[index, column]
            if index != column:
                if 0.7 <= abs(value) < 1:
                    strength = "strong"
                elif 0.4 <= abs(value) < 0.7:
                    strength = "moderate"
                elif 0.1 <= abs(value) < 0.4:
                    strength = "weak"
                else:
                    strength = "no or negligible"
                direction = "positive" if value > 0 else "negative"
                print(f"The correlation between {index} and {column} is {direction} and {strength}.")

def standardize_data(dataframe):
    scaler = StandardScaler()
    return scaler.fit_transform(dataframe)

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

def evaluate_silhouette(data, clusters):
    score = silhouette_score(data, clusters)
    if score < 0.25:
        evaluation = "No substantial structure has been found"
    elif 0.25 <= score < 0.5:
        evaluation = "A structure was found, but there's room for improvement"
    else:
        evaluation = "The structure is strong and clear"
    print(f"\nSilhouette Score: {score}, Evaluation: {evaluation}")

def describe_clusters(dataframe, clusters):
    dataframe['Cluster'] = clusters
    description = dataframe.groupby('Cluster').describe()
    print("\nCluster Description:")
    print(description)

def plot_3d_scatter(dataframe, clusters, n_clusters):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 设置全局字体大小和字体为 Arial
    plt.rcParams['font.size'] = 24
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    color_map = {0: '#134F85', 1: '#DBA11C', 2: '#FFEDCB'}
    colors = [color_map[cluster] for cluster in clusters]

    ax.scatter(dataframe['Depression'], dataframe['Anxiety'], dataframe['Stress'], c=colors, alpha=0.8, s=50)
    ax.set_xlabel("Depression Score")
    ax.set_ylabel("Anxiety Score")
    ax.set_zlabel("Pressure Score")
    # plt.title(f"Scatter Plot for clusters")
    plt.savefig(f'../draw/scatter_2clusters.png', dpi=1200, transparent=True)

def main():
    dataframe = pd.read_csv('../data/baseinfo.csv')
    data_for_analysis = dataframe[['Depression', 'Anxiety', 'Stress']]
    standardized_data = standardize_data(data_for_analysis)

    compute_correlation(data_for_analysis)

    cluster_ids = {'cust_id': dataframe['cust_id']}

    for n_clusters in [3]:
        clusters = kmeans_clustering(standardized_data, n_clusters)
        evaluate_silhouette(standardized_data, clusters)
        describe_clusters(dataframe, clusters)
        dataframe[f'Cluster_{n_clusters}'] = clusters
        cluster_ids[f'Cluster_{n_clusters}'] = clusters
        plot_3d_scatter(data_for_analysis, clusters, n_clusters)

    pd.DataFrame(cluster_ids).to_csv('../data/clusters.csv', index=False)

if __name__ == "__main__":
    main()
