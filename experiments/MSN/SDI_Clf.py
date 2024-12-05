import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import argparse
import torch.optim as optim
from MSN import MSN
import numpy as np
from sklearn.preprocessing import StandardScaler

class ImageDataset(Dataset):
    def __init__(self, data, root_dir, label_col, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.label_col = label_col
        self.transform = transform
        self.valid_ids = self._get_valid_ids()

    def _get_valid_ids(self):
        valid_ids = []
        for idx in range(len(self.data)):
            img_id = self.data.iloc[idx, 0]
            img_folder = os.path.join(self.root_dir, str(img_id))
            if os.path.exists(img_folder):
                valid_ids.append(idx)
        return valid_ids

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        data_idx = self.valid_ids[idx]
        img_id = self.data.iloc[data_idx, 0]
        label = self.data.iloc[data_idx][self.label_col]
        img_folder = os.path.join(self.root_dir, str(img_id))
        imgs = []
        for i in range(1, 8):  # Assuming 7 time frames as per your previous setting
            img_path = os.path.join(img_folder, f'{i}.jpg')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
        imgs = torch.stack(imgs)
        imgs = imgs.permute(1, 0, 2, 3)
        return imgs, label

def custom_collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

def calculate_sdi_and_select(data, factor_columns, scale_name, threshold):
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
        
        num_samples_to_keep = int(len(score_indices) * (1 - threshold))
        num_samples_to_keep = max(num_samples_to_keep, 1)
        
        selected_score_indices = score_indices[np.argsort(score_distances)[:num_samples_to_keep]]
        selected_indices.extend(selected_score_indices)
    
    return data.iloc[selected_indices]

def balance_dataset(dataset, label_col):
    labels = [item[1] for item in dataset]
    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    min_class_size = min(len(indices) for indices in label_to_indices.values())
    balanced_indices = []
    for indices in label_to_indices.values():
        balanced_indices.extend(np.random.choice(indices, min_class_size, replace=False))
    
    return torch.utils.data.Subset(dataset, balanced_indices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSN_Arg')
    parser.add_argument('--task', type=str, default='Anxiety', help='{depression, anxiety, dep_anx}')
    parser.add_argument('--task_form', type=str, default='c', help='{c for class task, r for rating task}')
    parser.add_argument('--lim', type=str, default='g', help='{g for LI_G, s for LI_S}')
    parser.add_argument('--backbone', type=str, default='res18', help='{res18, res34, res50}')
    parser.add_argument('--len_t', type=int, default=7, help='the length of input')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--model_id', type=str, default='define your model name')
    parser.add_argument('--log_path', type=str, default='log path, a txt file')
    parser.add_argument('--model_save_path', type=str, default='where the trained model is saved, a folder path')
    parser.add_argument('--csv_file', type=str, default='/home/user/xuxiao/FACES/data/label.csv')
    parser.add_argument('--dass_file', type=str, default='/home/user/xuxiao/FACES/data/dass21.csv')
    parser.add_argument('--root_dir', type=str, default='/home/user/xuxiao/FACES/data/process')
    parser.add_argument('--label_col', type=str, default='Anxiety')
    parser.add_argument('--sdi_threshold', type=float, default=0.2, help='SDI threshold for sample selection')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (3, 96, 96, 96)
    model = MSN(input_shape=input_shape).to(device)

    # 读取DASS数据并计算SDI
    dass_data = pd.read_csv(args.dass_file)
    dass_columns = {
        'Depression': ['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21'],
        'Anxiety': ['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20'],
        'Stress': ['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']
    }
    selected_data = calculate_sdi_and_select(dass_data, dass_columns[args.label_col], args.label_col, args.sdi_threshold)
    
    # 读取标签数据
    label_data = pd.read_csv(args.csv_file)
    
    # 合并选定的DASS数据和标签数据
    data = pd.merge(selected_data[['cust_id']], label_data, on='cust_id', how='inner')
    
    # 过滤和转换标签
    data = data[data[args.label_col] != 1]
    data.loc[data[args.label_col] == 2, args.label_col] = 1

    # 在数据预处理部分添加以下代码
    print(f"Unique labels in the dataset: {data[args.label_col].unique()}")

    # 确保标签从0开始
    data[args.label_col] = data[args.label_col] - data[args.label_col].min()

    print(f"Unique labels after adjustment: {data[args.label_col].unique()}")

    # 划分训练集、验证集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据加载器
    train_dataset = ImageDataset(train_data, args.root_dir, args.label_col, transform)
    val_dataset = ImageDataset(val_data, args.root_dir, args.label_col, transform)
    test_dataset = ImageDataset(test_data, args.root_dir, args.label_col, transform)

    # 在创建数据加载器之前，对数据集进行平衡
    train_dataset = balance_dataset(train_dataset, args.label_col)
    val_dataset = balance_dataset(val_dataset, args.label_col)
    test_dataset = balance_dataset(test_dataset, args.label_col)

    print(f"平衡后的训练样本数量: {len(train_dataset)}")
    print(f"平衡后的验证样本数量: {len(val_dataset)}")
    print(f"平衡后的测试样本数量: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    
    # 将损失函数改为 BCELoss
    Loss = nn.BCEWithLogitsLoss()

    # 修改模型的最后一层输出为单个神经元
    num_classes = 1
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 定义早停参数
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    # 训练循环
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)  # 确保标签是浮点型
    
            optimizer.zero_grad()
            out = model(imgs)
            out = out.squeeze()  # 确保输出维度与标签匹配
            loss = Loss(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().to(device)
                out = model(imgs)
                out = out.squeeze()
                loss = Loss(out, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_dataset)

        print(f'Epoch [{epoch+1}/{args.epoch}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.eval()
    test_loss = 0.0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            out = model(imgs)
            out = out.squeeze()
            loss = Loss(out, labels)
            test_loss += loss.item() * imgs.size(0)
            pred = (out > 0.5).float()  # 使用0.5作为阈值进行二分类
            correct += torch.sum(pred == labels)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    test_loss /= len(test_dataset)
    test_acc = correct / len(test_dataset)

    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    # 计算并打印混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f'Confusion Matrix:\n{conf_matrix}')

    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')
    print(f'Accuracy: {acc:.3f}, F1-score: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}')
