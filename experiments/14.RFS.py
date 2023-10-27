import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import *
import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np

def load_and_process_data(scale, category):
    label = pd.read_csv('../data/label.csv')[['cust_id', scale]]
    
    # Process data based on category
    if category == 2:
        label[scale] = label[scale].replace({1: 1, 2: 1})
    elif category == 3:
        label = label[label[scale] != 1]
        label[scale] = label[scale].replace({2: 1})
    else:
        raise ValueError("Invalid category value, should be 2 or 3.")
    
    features = pd.read_csv('../data/features.csv')
    data = pd.merge(label, features, left_on='cust_id', right_on='ID', how='left').drop(columns=['ID', 'cust_id'])
    data = data.rename(columns={scale: 'label'})
    X = data.drop(columns=['label']).values
    y = data['label'].values
    
    return X, y

def downsample_data(X, y):
    np.random.seed(0)

    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("The input labels do not contain exactly two classes.")
    
    # 获取每一类的索引
    indices_0 = np.where(y == unique_labels[0])[0]
    indices_1 = np.where(y == unique_labels[1])[0]
    
    # 确定下采样的数量
    sample_count = min(len(indices_0), len(indices_1))
    
    # 从每个类别中随机选择样本
    sampled_indices_0 = np.random.choice(indices_0, sample_count, replace=False)
    sampled_indices_1 = np.random.choice(indices_1, sample_count, replace=False)
    
    # 合并下采样的索引
    combined_indices = np.concatenate([sampled_indices_0, sampled_indices_1])
    
    # 使用这些索引返回下采样的 X 和 y
    return X[combined_indices], y[combined_indices]

def feature_selection(X, y, feature_names, alpha=0.01):
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("The input labels do not contain exactly two classes.")
    
    significant_features = []
    p_values_features = []

    for col in range(X.shape[1]):
        feature_values_class_0 = X[y == unique_labels[0], col]
        feature_values_class_1 = X[y == unique_labels[1], col]

        _, p_value = mannwhitneyu(feature_values_class_0, feature_values_class_1, alternative='two-sided')

        p_values_features.append((feature_names[col], p_value))

        if p_value < alpha:
            significant_features.append(col)
    
    X_significant = X[:, significant_features]
    
    # Return feature names instead of indices
    top_indices = np.argsort([p_value for _, p_value in p_values_features])
    top_features_p_values = [(feature_names[index], p_values_features[index][1]) for index in top_indices]
    
    # Aggregate feature p-values
    aggregated_features = aggregate_features(top_features_p_values)

    return X_significant, aggregated_features

def aggregate_features(feature_list):
    feature_dict = {}
    for feature, value in feature_list:
        # Remove content after the last underscore
        general_feature = '_'.join(feature.split('_')[:])
        
        # Aggregate values by feature category
        if general_feature in feature_dict:
            feature_dict[general_feature] += value
        else:
            feature_dict[general_feature] = value
            
    # Sort features by importance value
    aggregated_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
    
    return aggregated_features

def standardize_data(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized

def train_random_forest(X, y, n_estimators=300):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    rf.fit(X, y)
    return rf

def get_top_features_from_rf(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Get indices of top 30 features
    top_features = [(feature_names[i], importances[i]) for i in indices]
    
    # Aggregate feature importances
    aggregated_features = aggregate_features(top_features)
    
    return aggregated_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate machine learning models.')
    parser.add_argument('--scale', type=str, required=True, help='Scale name for evaluation.')

    args = parser.parse_args()

    scale = args.scale


    feature_names = pd.read_csv('../data/features.csv').columns.tolist()
    X, y = load_and_process_data(scale, 3)
    X, y = downsample_data(X, y)
    X, u_test_res = feature_selection(X, y, feature_names)
    X = standardize_data(X)
    rf_model = train_random_forest(X, y)

    top_rf_features = get_top_features_from_rf(rf_model, feature_names)
    # print("Top 10 features based on U-test:", u_test_res[:20])
    print("Top 10 features based on RandomForest:", top_rf_features[:50])
    pd.DataFrame(top_rf_features).to_csv(f'../results/features/{scale}.csv', index=False, header=['Feature', 'Value'])
