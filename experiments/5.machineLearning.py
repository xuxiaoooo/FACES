import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu
from sklearn.utils import resample
from xgboost import XGBClassifier
from collections import defaultdict
from statistics import mean, stdev

def load_data(scale, label_path, feature_path):
    df_label = pd.read_csv(label_path)[['cust_id', scale]]
    df_label = df_label.loc[df_label[scale] != 1]
    df_label[scale].replace(2, 1, inplace=True)
    df_features = pd.read_csv(feature_path)
    merged_df = pd.merge(df_features, df_label, left_on='ID', right_on='cust_id', how='inner')
    X = merged_df.drop(columns=['cust_id', 'ID', scale])
    y = merged_df[scale]
    return X, y

# 原始的基础分类器
base_classifiers = [
    AdaBoostClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(tree_method='gpu_hist')
]

# 对应的 Bagging 版本
bagging_classifiers = [BaggingClassifier(base_estimator=clf, n_estimators=10, random_state=42) for clf in base_classifiers]

# 所有分类器（包括原始和 Bagging 版本）
all_classifiers = base_classifiers + bagging_classifiers

# 原始分类器的参数网格
param_grids_base = [
    {'n_estimators': [10], 'learning_rate': [0.01]},
    {'n_estimators': [10], 'max_depth': [5]},
    {'n_estimators': [10], 'max_depth': [5]},
    {'n_estimators': [10], 'max_depth': [5], 'learning_rate': [0.01]}
]

# Bagging 分类器的参数网格（如果需要的话，你可以自定义这些）
param_grids_bagging = [
    {'base_estimator__n_estimators': [10], 'base_estimator__learning_rate': [0.01]},
    {'base_estimator__n_estimators': [10], 'base_estimator__max_depth': [5]},
    {'base_estimator__n_estimators': [10], 'base_estimator__max_depth': [5]},
    {'base_estimator__n_estimators': [10], 'base_estimator__max_depth': [5], 'base_estimator__learning_rate': [0.01]}
]

# 所有参数网格（包括原始和 Bagging 版本）
all_param_grids = param_grids_base + param_grids_bagging

def random_down_sampling(X, y, random_state):
    df = pd.concat([X, y], axis=1)
    # 分离多数和少数类
    class_counts = df[y.name].value_counts()
    majority_class_label = class_counts.idxmax()
    minority_class_label = class_counts.idxmin()
    majority_class = df[df[y.name] == majority_class_label]
    minority_class = df[df[y.name] == minority_class_label]
    # 下采样多数类
    majority_downsampled = resample(majority_class, 
                                    replace=False,  # 不放回抽样
                                    n_samples=len(minority_class),  # 将多数类样本数减少到与少数类一样
                                    random_state=random_state)  # 可重现的结果
    # 合并少数类和下采样后的多数类
    downsampled_df = pd.concat([majority_downsampled, minority_class])
    # 分离特征和标签
    X_downsampled = downsampled_df.drop(y.name, axis=1).reset_index(drop=True)
    y_downsampled = downsampled_df[y.name].reset_index(drop=True)
    return X_downsampled, y_downsampled

def u_test(X, y):
    significant_features = []
    p_values = {}
    for feature in X.columns:
        group1 = X[y == 0][feature]
        group2 = X[y == 1][feature]
        _, p_value = mannwhitneyu(group1, group2)
        if p_value < 0.01:
            significant_features.append(feature)
            p_values[feature] = p_value
    return significant_features, p_values

def z_score_standardization(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return pd.DataFrame(X_standardized, columns=X.columns)

def perform_pca(X):
    pca = PCA(n_components=0.99)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.n_components_, pca.explained_variance_ratio_

def evaluate_classifier(clf, X_train, y_train, X_test, y_test):    
    metrics_dict = {}    
    clf.fit(X_train, y_train)    
    y_pred = clf.predict(X_test)    
    y_proba = clf.predict_proba(X_test)[:, 1]    
    metrics_dict['ACC'] = round(accuracy_score(y_test, y_pred), 3)    
    metrics_dict['F1'] = round(f1_score(y_test, y_pred), 3)    
    metrics_dict['recall'] = round(recall_score(y_test, y_pred), 3)    
    metrics_dict['precision'] = round(precision_score(y_test, y_pred), 3)    
    metrics_dict['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()    
    fpr, tpr, _ = roc_curve(y_test, y_proba)    
    metrics_dict['roc_auc'] = round(auc(fpr, tpr), 3)    
    metrics_dict['fpr'] = [round(i, 3) for i in fpr.tolist()]
    metrics_dict['tpr'] = [round(i, 3) for i in tpr.tolist()]
    return metrics_dict

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

def main(scale, random_state, types):
    final_results = defaultdict(lambda: defaultdict(list))
    X, y = load_data(scale, f'../data/label_{types}.csv', '../data/features.csv')
    # 1. 随机下采样
    strat_time = time.time()
    X_downsampled, y_downsampled = random_down_sampling(X, y, random_state)
    end_time = time.time()
    print(f'Random Downsampling Elapsed time: {end_time - strat_time:.2f} s')
    # 2. U-检验
    start_time = time.time()
    significant_features, p_values = u_test(X_downsampled, y_downsampled)
    final_results['U-test features'] = significant_features
    final_results['U-test p-values'] = p_values
    end_time = time.time()
    print(f'U-test Elapsed time: {end_time - start_time:.2f} s')
    # 3. Z-score 标准化
    start_time = time.time()
    X_standardized = z_score_standardization(X_downsampled[significant_features])
    end_time = time.time()
    print(f'Standardization Elapsed time: {end_time - start_time:.2f} s')
    # 4. PCA 降维
    start_time = time.time()
    X_pca, n_pca_components, pca_variance_ratio = perform_pca(X_standardized)
    final_results['PCA n_components'] = n_pca_components
    final_results['PCA variance ratio'] = pca_variance_ratio.tolist()
    end_time = time.time()
    print(f'PCA Elapsed time: {end_time - start_time:.2f} s')
    # 5. 机器学习
    start_time = time.time()
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    for base_clf, param_grid in zip(all_classifiers, all_param_grids):
        clf = GridSearchCV(base_clf, param_grid, cv=kf, scoring='roc_auc', n_jobs=-1)
        clf_name = f"{clf.estimator.base_estimator.__class__.__name__}_Bagging" if isinstance(clf.estimator, BaggingClassifier) else clf.estimator.__class__.__name__
        print(clf_name)
        bagging_results = defaultdict(list)
        for i, (train_index, test_index) in enumerate(kf.split(X_pca, y_downsampled)):
            X_train, X_test = X_pca[train_index], X_pca[test_index]
            y_train, y_test = y_downsampled[train_index], y_downsampled[test_index]
            
            metrics_dict = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
            
            # 检查是否有 feature_importances_ 属性
            if hasattr(clf.best_estimator_, 'feature_importances_'):
                feature_importances = clf.best_estimator_.feature_importances_
                feature_names = X.columns
                
                # 使用列表存储多个重要特征和它们的重要性
                metrics_dict['important_features'] = []
                metrics_dict['feature_importances'] = []
                for name, importance in zip(feature_names, feature_importances):
                    metrics_dict['important_features'].append(name)
                    metrics_dict['feature_importances'].append(importance)
                    
            for metric, value in metrics_dict.items():
                bagging_results[metric].append(value)
                
        # 这里将嵌套的 defaultdict 存储到 final_results 的 'bagging_results' 键中
        final_results['bagging_results'][clf_name] = bagging_results
    final_results = defaultdict_to_dict(final_results)
    end_time = time.time()
    print(f'Machine Learning Elapsed time: {end_time - start_time:.2f} s')
    # 6. 保存结果
    with open(f'../results/pkl/{types}_{scale}_results_{random_state%10}.pkl', 'wb') as f:
        pickle.dump(final_results, f)

if __name__ == '__main__':
    # main('Cluster', 1, 2)
    # main('Cluster', 22, 2)
    # main('Cluster', 333, 2)
    # main('Depression', 1, 2)
    # main('Depression', 22, 2)
    # main('Depression', 333, 2)
    # main('Anxiety', 1, 2)
    # main('Anxiety', 22, 2)
    # main('Anxiety', 333, 2)
    # main('Stress', 1, 2)
    # main('Stress', 22, 2)
    # main('Stress', 333, 2)
    main('Cluster', 1, 3)
    main('Cluster', 22, 3)
    main('Cluster', 333, 3)
    main('Depression', 1, 3)
    main('Depression', 22, 3)
    main('Depression', 333, 3)
    main('Anxiety', 1, 3)
    main('Anxiety', 22, 3)
    main('Anxiety', 333, 3)
    main('Stress', 1, 3)
    main('Stress', 22, 3)
    main('Stress', 333, 3)