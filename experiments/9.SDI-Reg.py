import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import time

def load_and_process_data():
    print("加载数据...")
    features = pd.read_csv('../data/features.csv')
    dass21_scores = pd.read_csv('../data/dass21.csv')
    if 'ID' not in features.columns:
        raise ValueError("'ID' column not found in features file")
    data = features.merge(dass21_scores, left_on='ID', right_on='cust_id', how='inner')
    data = data.drop(['cust_id'], axis=1)
    print(f"数据加载完成，总样本数：{len(data)}")
    return data

def calculate_sdi_and_select(data, factor_columns, scale_name, threshold):
    print(f"应用SDI，阈值：{threshold}")
    scaler = StandardScaler()
    X = data[factor_columns].values
    X_standardized = scaler.fit_transform(X)
    center_vector = np.mean(X_standardized, axis=0)
    distances = np.linalg.norm(X_standardized - center_vector, axis=1)
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    factor_scores = data[scale_name]
    selected_indices = []
    for score in factor_scores.unique():
        score_mask = factor_scores == score
        score_indices = np.where(score_mask)[0]
        score_distances = normalized_distances[score_mask]
        
        num_samples_to_keep = max(int(len(score_indices) * (1 - threshold)), 1)
        selected_score_indices = score_indices[np.argsort(score_distances)[:num_samples_to_keep]]
        selected_indices.extend(selected_score_indices)
    
    selected_data = data.iloc[selected_indices]
    print(f"SDI后的样本数：{len(selected_data)}")
    return selected_data

def standardize_and_pca(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    pca = PCA(n_components=0.99, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_standardized)
    return X_pca, pca.n_components_, np.sum(pca.explained_variance_ratio_)

def train_and_evaluate_models(X, y, feature_subset_name, factor_name, threshold):
    models = {
        "SVM": SVR(),
        "RF": RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for name, model in models.items():
        print(f"训练和评估 {name} 模型...")
        y_pred = cross_val_predict(model, X, y, cv=kf)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        results.append({
            'Factor': factor_name,
            'Feature Subset': feature_subset_name,
            'Model': name,
            'RMSE': round(rmse, 3),
            'MAE': round(mae, 3),
            'Threshold': threshold
        })
    
    return results

def main():
    start_time = time.time()
    data = load_and_process_data()
    factors = {
        'Depression': ['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21'],
        'Anxiety': ['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20'],
        'Stress': ['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']
    }
    
    feature_columns = [col for col in data.columns if col not in ['ID', 'Depression', 'Anxiety', 'Stress']]
    
    feature_regex_list = [
        '^\s+eye.*',
        '^\s+[p]ose.*',
        '^\s+[xXyYzZ].*',
        '^\s+AU.*',
        '^\s+gaze.*',
        '.*'  # This will evaluate all features
    ]
    
    all_results = []
    for threshold in np.arange(0.5, 0, -0.1):
        threshold = round(threshold, 2)
        print(f"\n处理阈值: {threshold}")
        
        for factor, dass_columns in factors.items():
            print(f"\n处理因子: {factor}")
            selected_data = calculate_sdi_and_select(data, dass_columns, factor, threshold)
            y = selected_data[factor].values
            
            for regex in feature_regex_list:
                X_filtered = selected_data[feature_columns].filter(regex=regex)
                
                feature_subset_name = regex.strip('^').replace('.*', '').replace('\s+', '').replace('[', '').replace(']', '').capitalize()
                if not feature_subset_name:
                    feature_subset_name = 'All Features'
                
                print(f"处理特征子集: {feature_subset_name}")
                print(f"特征数量: {X_filtered.shape[1]}")
                
                X_pca, n_components, explained_variance = standardize_and_pca(X_filtered)
                print(f"PCA后的特征数量: {X_pca.shape[1]}")
                
                results = train_and_evaluate_models(X_pca, y, feature_subset_name, factor, threshold)
                all_results.extend(results)
    
    results_df = pd.DataFrame(all_results)
    print("\n最终结果:")
    print(results_df)
    results_df.to_csv('sdi_threshold_results_regression_with_feature_subsets.csv', index=False)
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()
