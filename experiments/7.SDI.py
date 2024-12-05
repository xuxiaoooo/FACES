import pandas as pd
import numpy as np

def standardize(df, columns):
    return (df[columns] - df[columns].mean()) / df[columns].std()

def categorize_severity(df, column, bins, labels):
    df[column + '_Severity'] = pd.cut(df[column], bins=bins, labels=labels)

def calculate_center_vector(df, group_column, target_columns):
    groups = df.groupby(group_column)
    center_vectors = {}
    for group_name, group in groups:
        center_vector = group[target_columns].mean()
        center_vectors[group_name] = center_vector
    return center_vectors

def calculate_distances(df, group_column, target_columns, center_vectors):
    distances = []
    for index, row in df.iterrows():
        group = row[group_column]
        center_vector = center_vectors[group]
        distance = np.sqrt(((row[target_columns] - center_vector) ** 2).sum())
        distances.append(distance)
    return distances

def normalize_distances(distances):
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    return normalized_distances

def calculate_gdi(df, group_column, target_columns):
    center_vectors = calculate_center_vector(df, group_column, target_columns)
    distances = calculate_distances(df, group_column, target_columns, center_vectors)
    normalized_distances = normalize_distances(distances)
    df['distance'] = normalized_distances
    mean_distance = df.groupby(group_column)['distance'].mean()
    std_distance = df.groupby(group_column)['distance'].std()
    gdi = mean_distance + std_distance
    return gdi, df.groupby(group_column)['distance'].agg(['mean', 'std', 'min', 'max', 'count'])

def calculate_sdi(gdi, counts):
    weighted_sum = (gdi * counts).sum()
    total_count = counts.sum()
    sdi = weighted_sum / total_count
    return sdi

# 主要处理流程
df = pd.read_csv('../data/dass21.csv')

# 定义各维度的列
depression_columns = ['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21']
anxiety_columns = ['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20']
stress_columns = ['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']

# 标准化处理
df_standardized = df.copy()
all_columns = depression_columns + anxiety_columns + stress_columns
df_standardized[all_columns] = standardize(df, all_columns)

# 计算各维度总分
df_standardized['Depression'] = df_standardized[depression_columns].sum(axis=1)
df_standardized['Anxiety'] = df_standardized[anxiety_columns].sum(axis=1)
df_standardized['Stress'] = df_standardized[stress_columns].sum(axis=1)

# 对各维度进行严重程度分类
categorize_severity(df_standardized, 'Depression', [-1, 9, 13, 20, 27, float('inf')], ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
categorize_severity(df_standardized, 'Anxiety', [-1, 7, 9, 14, 19, float('inf')], ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
categorize_severity(df_standardized, 'Stress', [-1, 14, 18, 25, 33, float('inf')], ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])

# 计算各维度的SDI
for dimension, columns in [('Depression', depression_columns), ('Anxiety', anxiety_columns), ('Stress', stress_columns)]:
    # 基于总分的SDI
    gdi_score, summary_score = calculate_gdi(df_standardized, dimension, columns)
    sdi_score = calculate_sdi(gdi_score, summary_score['count'])
    print(f"\n{dimension} SDI (Same Score): {sdi_score:.4f}")
    print(summary_score)

    # 基于严重程度的SDI
    gdi_severity, summary_severity = calculate_gdi(df_standardized, dimension + '_Severity', columns)
    sdi_severity = calculate_sdi(gdi_severity, summary_severity['count'])
    print(f"\n{dimension} SDI (Same Severity): {sdi_severity:.4f}")
    print(summary_severity)

print("\nEvaluation Criteria:")
print("Value range [0, 1]. SDI value close to 0 indicates small discrepancy in subscale distribution; SDI value close to 1 indicates large discrepancy in subscale distribution.")
