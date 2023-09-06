import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

def load_from_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Cluster, Depression, Anxiety, Stress
types = '3_'
scale = 'Anxiety'
# 使用函数读取数据
df1 = load_from_pkl(f'../results/pkl/{types}{scale}_results_1.pkl')
df2 = load_from_pkl(f'../results/pkl/{types}{scale}_results_2.pkl')
df3 = load_from_pkl(f'../results/pkl/{types}{scale}_results_3.pkl')

# ['U-test features', 'U-test p-values', 'PCA n_components', 'PCA variance ratio', 'bagging_results']

# Utest
pvalues_list = [df1['U-test p-values'], df2['U-test p-values'], df3['U-test p-values']]
features_dict = defaultdict(list)
for pvalues in pvalues_list:
    for feature, pvalue in pvalues.items():
        try:
            pvalue = float(pvalue)  # 尝试转换为浮点数
            features_dict[feature].append(pvalue)
        except ValueError:
            print(f"Warning: could not convert p-value for feature {feature} to float. Skipping.")
for feature, pvalues in features_dict.items():
    features_dict[feature] = sum(pvalues) / len(pvalues)
result_df = pd.DataFrame(list(features_dict.items()), columns=['Name', 'Value'])
result_df = result_df.sort_values(by='Value')
print(result_df)
result_df.to_csv(f'../results/csv/{scale}_Utest_{types}.csv', index=False)

# PCA
# print(max(max(df1['PCA variance ratio']),max(df2['PCA variance ratio']),max(df3['PCA variance ratio'])))
# print(min(min(df1['PCA variance ratio']),min(df2['PCA variance ratio']),min(df3['PCA variance ratio'])))

# Bagging
# all_results = pd.DataFrame()
# names_base = ['AdaBoostClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'XGBClassifier']
# names_bagging = ['AdaBoostClassifier_Bagging', 'RandomForestClassifier_Bagging', 'ExtraTreesClassifier_Bagging', 'XGBClassifier_Bagging']
# for name in names_base:
#     c1 = df1['bagging_results'][name]
#     c2 = df2['bagging_results'][name]
#     c3 = df3['bagging_results'][name]

#     # ['ACC', 'F1', 'recall', 'precision', 'confusion_matrix', 'roc_auc', 'fpr', 'tpr', 'important_features', 'feature_importances']

#     acc = str(round(np.mean(c1['ACC']+c2['ACC']+c3['ACC']), 3)) + ' ± ' + str(round(np.std(c1['ACC']+c2['ACC']+c3['ACC']), 3))
#     f1 = str(round(np.mean(c1['F1']+c2['F1']+c3['F1']), 3)) + ' ± ' + str(round(np.std(c1['F1']+c2['F1']+c3['F1']), 3))
#     recall = str(round(np.mean(c1['recall']+c2['recall']+c3['recall']), 3)) + ' ± ' + str(round(np.std(c1['recall']+c2['recall']+c3['recall']), 3))
#     precision = str(round(np.mean(c1['precision']+c2['precision']+c3['precision']), 3)) + ' ± ' + str(round(np.std(c1['precision']+c2['precision']+c3['precision']), 3))
#     roc_auc = str(round(np.mean(c1['roc_auc']+c2['roc_auc']+c3['roc_auc']), 3)) + ' ± ' + str(round(np.std(c1['roc_auc']+c2['roc_auc']+c3['roc_auc']), 3))
#     max_idx = np.argmax(roc_auc)
#     if max_idx < len(c1['roc_auc']):
#         best_fpr = c1['fpr'][max_idx]
#         best_tpr = c1['tpr'][max_idx]
#     elif max_idx < len(c1['roc_auc']) + len(c2['roc_auc']):
#         best_fpr = c2['fpr'][max_idx - len(c1['roc_auc'])]
#         best_tpr = c2['tpr'][max_idx - len(c1['roc_auc'])]
#     else:
#         best_fpr = c3['fpr'][max_idx - len(c1['roc_auc']) - len(c2['roc_auc'])]
#         best_tpr = c3['tpr'][max_idx - len(c1['roc_auc']) - len(c2['roc_auc'])]
#     res = pd.DataFrame({
#         'Name': [name],
#         'Accuracy': [acc],
#         'F1 Score': [f1],
#         'Recall': [recall],
#         'Precision': [precision],
#         'ROC AUC': [roc_auc],
#         'Best FPR': [best_fpr],
#         'Best TPR': [best_tpr]
#     })
#     all_results = pd.concat([all_results, res], ignore_index=True)

# for name in names_bagging:
#     c1 = df1['bagging_results'][name]
#     c2 = df2['bagging_results'][name]
#     c3 = df3['bagging_results'][name]

#     acc = str(round(np.max(c1['ACC']+c2['ACC']+c3['ACC']), 3)) + ' ± ' + str(round(np.std(c1['ACC']+c2['ACC']+c3['ACC']), 3))
#     f1 = str(round(np.max(c1['F1']+c2['F1']+c3['F1']), 3)) + ' ± ' + str(round(np.std(c1['F1']+c2['F1']+c3['F1']), 3))
#     recall = str(round(np.max(c1['recall']+c2['recall']+c3['recall']), 3)) + ' ± ' + str(round(np.std(c1['recall']+c2['recall']+c3['recall']), 3))
#     precision = str(round(np.max(c1['precision']+c2['precision']+c3['precision']), 3)) + ' ± ' + str(round(np.std(c1['precision']+c2['precision']+c3['precision']), 3))
#     roc_auc = str(round(np.max(c1['roc_auc']+c2['roc_auc']+c3['roc_auc']), 3)) + ' ± ' + str(round(np.std(c1['roc_auc']+c2['roc_auc']+c3['roc_auc']), 3))
#     max_idx = np.argmax(roc_auc)
#     if max_idx < len(c1['roc_auc']):
#         best_fpr = c1['fpr'][max_idx]
#         best_tpr = c1['tpr'][max_idx]
#     elif max_idx < len(c1['roc_auc']) + len(c2['roc_auc']):
#         best_fpr = c2['fpr'][max_idx - len(c1['roc_auc'])]
#         best_tpr = c2['tpr'][max_idx - len(c1['roc_auc'])]
#     else:
#         best_fpr = c3['fpr'][max_idx - len(c1['roc_auc']) - len(c2['roc_auc'])]
#         best_tpr = c3['tpr'][max_idx - len(c1['roc_auc']) - len(c2['roc_auc'])]
#     res = pd.DataFrame({
#         'Name': [name],
#         'Accuracy': [acc],
#         'F1 Score': [f1],
#         'Recall': [recall],
#         'Precision': [precision],
#         'ROC AUC': [roc_auc],
#         'Best FPR': [best_fpr],
#         'Best TPR': [best_tpr]
#     })
#     all_results = pd.concat([all_results, res], ignore_index=True)

# all_results.to_csv(f'../results/csv/{types}{scale}_metrics.csv', index=False)

# # Important Features
# fe = pd.read_pickle(f'../results/old/{scale}_importance.pkl')
# accumulated_dict = defaultdict(lambda: {'value': 0, 'count': 0})

# # 输入的四个字典
# dict_list = [
#     fe['feature_importance_dict']['AdaBoostClassifier'],
#     fe['feature_importance_dict']['RandomForestClassifier'],
#     fe['feature_importance_dict']['ExtraTreesClassifier'],
#     fe['feature_importance_dict']['XGBClassifier']
# ]

# # 遍历每个字典，累加值和计数
# for d in dict_list:
#     for k, v in d.items():
#         accumulated_dict[k]['value'] += v
#         accumulated_dict[k]['count'] += 1
# average_dict = {k: v['value'] / v['count'] for k, v in accumulated_dict.items()}
# sorted_dict = {k: round(v, 3) for k, v in sorted(average_dict.items(), key=lambda item: item[1], reverse=True)}
# feature_df = pd.DataFrame(list(sorted_dict.items()), columns=['Feature', 'Value'])
# feature_df.to_csv(f'../results/csv/importance/{scale}.csv', index=False)
