import pandas as pd
import numpy as np

def standardize(df, columns):
    return (df[columns] - df[columns].mean()) / df[columns].std()

def categorize_severity(df, column, bins, labels):
    df[column + '_Severity'] = pd.cut(df[column], bins=bins, labels=labels)

def calculate_center_vector(df, group_column, target_columns):
    return df.groupby(group_column)[target_columns].mean()

def calculate_distances(df, group_column, target_columns, center_vectors):
    distances = []
    for group_name, group in df.groupby(group_column):
        center_vector = center_vectors.loc[group_name]
        group_distances = np.sqrt(((group[target_columns] - center_vector) ** 2).sum(axis=1))
        distances.extend(group_distances)
    return distances

def normalize_distances(distances):
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return (distances - min_dist) / (max_dist - min_dist)

def calculate_gdi_sdi(df, group_column, target_columns, remove_percentage=0):
    center_vectors = calculate_center_vector(df, group_column, target_columns)
    distances = calculate_distances(df, group_column, target_columns, center_vectors)
    normalized_distances = normalize_distances(distances)
    
    if remove_percentage > 0:
        threshold = np.percentile(normalized_distances, 100 - remove_percentage)
        mask = normalized_distances <= threshold
        df_filtered = df[mask]
        normalized_distances = normalized_distances[mask]
    else:
        df_filtered = df
    
    df_filtered['distance'] = normalized_distances
    group_stats = df_filtered.groupby(group_column)['distance'].agg(['mean', 'std', 'count'])
    gdi = group_stats['mean'] + group_stats['std']
    sdi = (gdi * group_stats['count']).sum() / group_stats['count'].sum()
    return sdi

# 主要处理流程
df = pd.read_csv('../data/dass21.csv')

# 定义各维度的列
depression_columns = ['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21']
anxiety_columns = ['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20']
stress_columns = ['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']

# 计算各维度总分
df['Depression'] = df[depression_columns].sum(axis=1)
df['Anxiety'] = df[anxiety_columns].sum(axis=1)
df['Stress'] = df[stress_columns].sum(axis=1)

# 对各维度进行严重程度分类
categorize_severity(df, 'Depression', [-1, 9, 13, 20, 27, float('inf')], ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
categorize_severity(df, 'Anxiety', [-1, 7, 9, 14, 19, float('inf')], ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
categorize_severity(df, 'Stress', [-1, 14, 18, 25, 33, float('inf')], ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])

# 创建结果表格
results = []

# 计算各维度的SDI
for dimension, columns in [('Depression', depression_columns), ('Anxiety', anxiety_columns), ('Stress', stress_columns)]:
    # 基于总分的SDI
    for remove_percentage in [0, 10, 20, 30, 40, 50]:
        sdi_score = calculate_gdi_sdi(df, dimension, columns, remove_percentage)
        results.append({
            'Dimension': dimension,
            'Grouping': 'Total Score',
            'Removed %': remove_percentage,
            'SDI': sdi_score
        })
    
    # 基于严重程度的SDI
    for remove_percentage in [0, 10, 20, 30, 40, 50]:
        sdi_severity = calculate_gdi_sdi(df, dimension + '_Severity', columns, remove_percentage)
        results.append({
            'Dimension': dimension,
            'Grouping': 'Severity',
            'Removed %': remove_percentage,
            'SDI': sdi_severity
        })

# 创建DataFrame并显示结果
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\nEvaluation Criteria:")
print("SDI value range [0, 1]. SDI value close to 0 indicates small discrepancy in subscale distribution; SDI value close to 1 indicates large discrepancy in subscale distribution.")