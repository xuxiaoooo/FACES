import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import resample
from scipy.interpolate import interp1d

def load_data(scale, label_path, feature_path):
    df_label = pd.read_csv(label_path)[['cust_id', scale]]
    df_label = df_label.loc[df_label[scale] != 1]
    df_label[scale].replace(2, 1, inplace=True)
    df_features = pd.read_csv(feature_path)
    merged_df = pd.merge(df_features, df_label, left_on='ID', right_on='cust_id', how='inner')
    X = merged_df.drop(columns=['cust_id', 'ID', scale])
    y = merged_df[scale]
    return X, y

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

# 读取数据
scale = 'Cluster_3'
X, y = load_data(scale, '../data/label.csv', '../data/features.csv')
X, y = random_down_sampling(X, y, 42)
X = X.values
y = y.values

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        # 128 filters of kernel size 7x1
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # 128 filters of kernel size 5x1
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # 64 filters of kernel size 3x1
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # Channel-level average pooling to obtain a 1-D feature from each feature map
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        # Dropout layer with p=0.5
        self.dropout = nn.Dropout(0.05)
        # Output layer with one neuron
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# 评估函数
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float().view(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    
    return accuracy, f1, recall, precision, roc_auc, fpr, tpr

# StratifiedKFold
skf = StratifiedKFold(n_splits=5)
metrics_values = []  # 用于存储5个评价指标
all_fpr = []  # 存储所有的fpr序列
all_tpr = []  # 存储所有的tpr序列

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 转换为 PyTorch DataLoader
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train).unsqueeze(1), torch.FloatTensor(y_train).unsqueeze(1))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test).unsqueeze(1), torch.FloatTensor(y_test).unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = Net(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0000001)

    # 训练模型
    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    metrics = evaluate(model, test_loader)
    metrics_values.append(metrics[:5])  # 保存5个评价指标
    all_fpr.append(metrics[5])  # 保存fpr
    all_tpr.append(metrics[6])  # 保存tpr

# 计算五个评价指标的平均值和标准差
metrics_np = np.array(metrics_values)
mean_metrics = metrics_np.mean(axis=0)
std_metrics = metrics_np.std(axis=0)

print(f"Accuracy: {mean_metrics[0]:.3f} ± {std_metrics[0]:.3f}")
print(f"F1: {mean_metrics[1]:.3f} ± {std_metrics[1]:.3f}")
print(f"Recall: {mean_metrics[2]:.3f} ± {std_metrics[2]:.3f}")
print(f"Precision: {mean_metrics[3]:.3f} ± {std_metrics[3]:.3f}")
print(f"ROC_AUC: {mean_metrics[4]:.3f} ± {std_metrics[4]:.3f}")

# 插值处理 FPR 和 TPR
base_fpr = np.linspace(0, 1, 101)  # 常见的方法是使用101个点，从0到1
tpr_interp = []

for fpr, tpr in zip(all_fpr, all_tpr):
    interp_func = interp1d(fpr, tpr, kind='linear', fill_value=(0, 1), bounds_error=False)
    tpr_interp.append(interp_func(base_fpr))

mean_tpr = np.mean(tpr_interp, axis=0)

# 保存均值到 CSV 文件
# roc_df = pd.DataFrame({'FPR': base_fpr, 'TPR': mean_tpr})
# roc_df.to_csv(f'../results/csv/metrics/CNN_{scale}_roc.csv', index=False)
