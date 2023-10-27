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
import pickle

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

def feature_selection(X, y, alpha=0.01):
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("The input labels do not contain exactly two classes.")
    
    significant_features = []
    p_values = []

    # 对每一列特征进行 U 检验
    for col in range(X.shape[1]):
        feature_values_class_0 = X[y == unique_labels[0], col]
        feature_values_class_1 = X[y == unique_labels[1], col]

        # 使用 Mann-Whitney U 检验
        _, p_value = mannwhitneyu(feature_values_class_0, feature_values_class_1, alternative='two-sided')

        p_values.append(p_value)

        if p_value < alpha:
            significant_features.append(col)
    
    # 获取显著差异的特征矩阵
    X_significant = X[:, significant_features]
    
    # 获取按p值排序的前10个特征及其p值
    top_indices = np.argsort(p_values)[:10]
    top_features_p_values = [(index, p_values[index]) for index in top_indices]
    
    return X_significant, top_features_p_values

def standardize_data(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized

def apply_pca(X_standardized, explained_variance_ratio_threshold=0.99):
    pca = PCA(n_components=explained_variance_ratio_threshold, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_standardized)
    num_components = pca.n_components_
    cumulative_explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    
    return X_pca, num_components, cumulative_explained_variance_ratio

def get_classifiers_and_params():
    # AdaBoost
    ada = AdaBoostClassifier()
    ada_params = {
        'n_estimators': [150, 300],
        'learning_rate': [0.01, 0.1]
    }

    # RandomForest
    rf = RandomForestClassifier()
    rf_params = {
        'n_estimators': [150, 300],
        'max_depth': [None, 10]
    }

    # ExtraTrees
    et = ExtraTreesClassifier()
    et_params = {
        'n_estimators': [150, 300],
        'max_depth': [None, 10]
    }

    # XGBoost
    xgb = XGBClassifier(tree_method='gpu_hist')
    xgb_params = {
        'n_estimators': [150, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6]
    }

    classifiers = [ada, rf, et, xgb]
    params = [ada_params, rf_params, et_params, xgb_params]

    # Bagging versions
    ada_bagged = BaggingClassifier(base_estimator=AdaBoostClassifier())
    rf_bagged = BaggingClassifier(base_estimator=RandomForestClassifier())
    et_bagged = BaggingClassifier(base_estimator=ExtraTreesClassifier())
    xgb_bagged = BaggingClassifier(base_estimator=XGBClassifier(tree_method='gpu_hist'))

    param_grids_bagging = [
        {'base_estimator__n_estimators': [300], 'base_estimator__learning_rate': [0.001]},
        {'base_estimator__n_estimators': [300], 'base_estimator__max_depth': [100]},
        {'base_estimator__n_estimators': [300], 'base_estimator__max_depth': [100]},
        {'base_estimator__n_estimators': [300], 'base_estimator__max_depth': [100], 'base_estimator__learning_rate': [0.001]}
    ]

    classifiers.extend([ada_bagged, rf_bagged, et_bagged, xgb_bagged])
    params.extend(param_grids_bagging)

    return classifiers, params

def get_selected_classifiers_and_params(model_abbreviation):
    classifiers, params = get_classifiers_and_params()
    
    model_mapping = {
        "ada": 0,
        "rf": 1,
        "et": 2,
        "xgb": 3,
        "ada_b": 4,
        "rf_b": 5,
        "et_b": 6,
        "xgb_b": 7
    }

    index = model_mapping.get(model_abbreviation)
    if index is None:
        raise ValueError(f"Model abbreviation {model_abbreviation} is not recognized.")

    return [classifiers[index]], [params[index]]

def evaluate_model(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rocauc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC AUC': rocauc,
        'Confusion Matrix': conf_matrix,
        'FPR': fpr,
        'TPR': tpr
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate machine learning models.')
    parser.add_argument('--model', type=str, required=True, help='Abbreviation of the machine learning model.')
    parser.add_argument('--scale', type=str, required=True, help='Scale name for evaluation.')
    parser.add_argument('--category', type=int, choices=[2, 3], required=True, help='Choose 2 or 3 for classification.')
    # python 5.machineL.py --model ada --scale Depression --category 2

    args = parser.parse_args()

    scale = args.scale
    model_abbreviation = args.model
    category = args.category

    X, y = load_and_process_data(scale, category)
    X, y = downsample_data(X, y)
    X, u_test_res = feature_selection(X, y)
    X = standardize_data(X)
    X, pca_components, pca_variance = apply_pca(X)

    classifiers, params = get_selected_classifiers_and_params(model_abbreviation)
    for clf, param in zip(classifiers, params):
        results = {}
        results['u_test'] = u_test_res
        results['pca_components'] = pca_components
        results['pca_variance'] = pca_variance
        
        grid = GridSearchCV(clf, param, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X, y)

        y_pred = cross_val_predict(grid.best_estimator_, X, y, cv=5)
        y_prob = cross_val_predict(grid.best_estimator_, X, y, cv=5, method='predict_proba')[:, 1]

        model_results = evaluate_model(y, y_pred, y_prob)
        if isinstance(model_results, dict):
            model_results = pd.Series(model_results)

        results[model_abbreviation] = model_results

        with open(f"../results/clfs/{scale}_{model_abbreviation}_{category}.pkl", "wb") as f:
            pickle.dump(results, f)