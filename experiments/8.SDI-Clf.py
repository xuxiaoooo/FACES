import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from scipy.stats import mannwhitneyu

def load_and_process_data():
    features = pd.read_csv('../data/features.csv')
    dass21_scores = pd.read_csv('../data/dass21.csv')
    if 'ID' not in features.columns:
        raise ValueError("'ID' 列在特征文件中未找到")
    data = features.merge(dass21_scores, left_on='ID', right_on='cust_id', how='inner')
    data = data.drop(['cust_id'], axis=1)
    return data

def calculate_sdi_and_select(data, factor_columns, scale_name, threshold):
    scaler = StandardScaler()
    X = data[factor_columns].values
    X_standardized = scaler.fit_transform(X)
    center_vector = np.mean(X_standardized, axis=0)
    distances = np.linalg.norm(X_standardized - center_vector, axis=1)
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    # 获取因子得分
    factor_scores = data[scale_name]
    
    # 为每个唯一的得分值应用SDI
    selected_indices = []
    for score in factor_scores.unique():
        score_mask = factor_scores == score
        score_indices = np.where(score_mask)[0]
        score_distances = normalized_distances[score_mask]
        
        num_samples_to_keep = int(len(score_indices) * (1 - threshold))
        num_samples_to_keep = max(num_samples_to_keep, 1)  # 确保至少保留1个样本
        
        selected_score_indices = score_indices[np.argsort(score_distances)[:num_samples_to_keep]]
        selected_indices.extend(selected_score_indices)
    
    return data.iloc[selected_indices]

def balance_data(X, y):
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)
    X_minority = X[y == minority_class]
    X_majority = X[y == majority_class]
    y_minority = y[y == minority_class]
    y_majority = y[y == majority_class]
    X_majority_downsampled, y_majority_downsampled = resample(X_majority, y_majority, replace=False, n_samples=len(X_minority), random_state=42)
    X_balanced = np.vstack((X_minority, X_majority_downsampled))
    y_balanced = np.hstack((y_minority, y_majority_downsampled))
    return X_balanced, y_balanced

def standardize_and_pca(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    pca = PCA(n_components=0.99, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_standardized)
    return X_pca, pca.n_components_, np.sum(pca.explained_variance_ratio_)

def train_and_evaluate_models(X, y):
    rf = RandomForestClassifier(random_state=42)
    svm = SVC(probability=True, random_state=42)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 随机森林网格搜索
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search_rf.fit(X, y)
    
    # SVM网格搜索
    grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search_svm.fit(X, y)
    
    # 交叉验证预测
    y_pred_rf = cross_val_predict(grid_search_rf.best_estimator_, X, y, cv=cv)
    y_pred_svm = cross_val_predict(grid_search_svm.best_estimator_, X, y, cv=cv)
    
    results_rf = {
        'Accuracy': (np.mean(grid_search_rf.cv_results_['mean_test_score']), np.std(grid_search_rf.cv_results_['mean_test_score'])),
        'Precision': (precision_score(y, y_pred_rf), np.std([precision_score(y[train], y_pred_rf[train]) for train, _ in cv.split(X, y)])),
        'Recall': (recall_score(y, y_pred_rf), np.std([recall_score(y[train], y_pred_rf[train]) for train, _ in cv.split(X, y)])),
        'F1 Score': (f1_score(y, y_pred_rf), np.std([f1_score(y[train], y_pred_rf[train]) for train, _ in cv.split(X, y)]))
    }
    
    results_svm = {
        'Accuracy': (np.mean(grid_search_svm.cv_results_['mean_test_score']), np.std(grid_search_svm.cv_results_['mean_test_score'])),
        'Precision': (precision_score(y, y_pred_svm), np.std([precision_score(y[train], y_pred_svm[train]) for train, _ in cv.split(X, y)])),
        'Recall': (recall_score(y, y_pred_svm), np.std([recall_score(y[train], y_pred_svm[train]) for train, _ in cv.split(X, y)])),
        'F1 Score': (f1_score(y, y_pred_svm), np.std([f1_score(y[train], y_pred_svm[train]) for train, _ in cv.split(X, y)]))
    }
    
    return results_rf, results_svm

def get_binary_labels_for_dass_factor(df, factor_name):
    if factor_name == 'Depression':
        return (df['Depression'] > 9).astype(int)
    elif factor_name == 'Anxiety':
        return (df['Anxiety'] > 7).astype(int)
    elif factor_name == 'Stress':
        return (df['Stress'] > 14).astype(int)
    else:
        raise ValueError("未知的因子名称！")

def main():
    data = load_and_process_data()
    factors = {
        'Depression': ['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21'],
        'Anxiety': ['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20'],
        'Stress': ['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']
    }
    feature_columns = [col for col in data.columns if col not in ['ID', 'Depression', 'Anxiety', 'Stress']]
    all_results = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        threshold = round(threshold, 2)
        print(f"\n处理阈值: {threshold}")
        results = []
        for factor, dass_columns in factors.items():
            print(f"\n处理 {factor}")
            selected_data = calculate_sdi_and_select(data, dass_columns, factor, threshold)
            X = selected_data[feature_columns].values
            y = get_binary_labels_for_dass_factor(selected_data, factor)
            print(f"当前处理的样本数量: {len(selected_data)}")
            
            # 在此处添加数据平衡步骤
            X_balanced, y_balanced = balance_data(X, y)
            print(f"平衡后的样本数量: {len(X_balanced)}")
            print(f"平衡后的类别分布: {np.bincount(y_balanced)}")

            # 标准化和PCA
            X_pca, n_components, explained_variance = standardize_and_pca(X_balanced)
            
            # 训练和评估模型
            results_rf, results_svm = train_and_evaluate_models(X_pca, y_balanced)
            results.append({
                'Threshold': threshold,
                'Factor': factor,
                'RF Accuracy': f"{results_rf['Accuracy'][0]:.2f} ± {results_rf['Accuracy'][1]:.2f}",
                'RF Precision': f"{results_rf['Precision'][0]:.2f} ± {results_rf['Precision'][1]:.2f}",
                'RF Recall': f"{results_rf['Recall'][0]:.2f} ± {results_rf['Recall'][1]:.2f}",
                'RF F1 Score': f"{results_rf['F1 Score'][0]:.2f} ± {results_rf['F1 Score'][1]:.2f}",
                'SVM Accuracy': f"{results_svm['Accuracy'][0]:.2f} ± {results_svm['Accuracy'][1]:.2f}",
                'SVM Precision': f"{results_svm['Precision'][0]:.2f} ± {results_svm['Precision'][1]:.2f}",
                'SVM Recall': f"{results_svm['Recall'][0]:.2f} ± {results_svm['Recall'][1]:.2f}",
                'SVM F1 Score': f"{results_svm['F1 Score'][0]:.2f} ± {results_svm['F1 Score'][1]:.2f}"
            })
        all_results.extend(results)
    results_df = pd.DataFrame(all_results)
    print("\n最终结果:")
    print(results_df)
    results_df.to_csv('sdi_threshold_results_classification.csv', index=False)

if __name__ == '__main__':
    main()
