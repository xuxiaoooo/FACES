import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
from LI_FPN import LI_FPN
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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
            # label = self.data.iloc[idx][self.label_col]
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
        for i in range(1, 8):
            img_path = os.path.join(img_folder, f'{i}.jpg')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
        imgs = torch.stack(imgs)
        # print(label)
        return imgs, label

def custom_collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LI_FPN_Arg')
    parser.add_argument('--task', type=str, default='anxiety', help='{depression, anxiety, dep_anx}')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--task_form', type=str, default='c', help='{c for class task, r for rating task}')
    parser.add_argument('--lim', type=str, default='g', help='{g for LI_G, s for LI_S}')
    parser.add_argument('--backbone', type=str, default='res18', help='{res18, res34, res50}')
    parser.add_argument('--len_t', type=int, default=7, help='the length of input')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--model_id', type=str, default='define your model name')
    parser.add_argument('--log_path', type=str, default='log path, a txt file')
    parser.add_argument('--model_save_path', type=str, default='where the trained model is saved, a folder path')
    parser.add_argument('--csv_file', type=str, default='/home/user/xuxiao/FACES/data/label.csv')
    parser.add_argument('--root_dir', type=str, default='/home/user/xuxiao/FACES/data/process')
    parser.add_argument('--label_col', type=str, default='Cluster_3')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LI_FPN(class_num=args.class_num,
                   task_form=args.task_form,
                   lim=args.lim,
                   backbone=args.backbone,
                   len_t=args.len_t,
                   pretrain=args.pretrain).to(device)

    # 读取CSV文件
    data = pd.read_csv(args.csv_file)
    
    data = data[data[args.label_col] != 1]
    data.loc[data[args.label_col] == 2, args.label_col] = 1

    # 进行降采样
    class_0 = data[data[args.label_col] == 0]
    class_1 = data[data[args.label_col] == 1]

    # 计算降采样后的样本数量
    n_samples = min(len(class_0), len(class_1))

    # 随机选择n_samples个样本
    class_0_downsampled = class_0.sample(n=n_samples, random_state=42)
    class_1_downsampled = class_1.sample(n=n_samples, random_state=42)

    # 合并降采样后的数据
    balanced_data = pd.concat([class_0_downsampled, class_1_downsampled]).reset_index(drop=True)

    # 划分训练集、验证集和测试集
    train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=1/2, random_state=42)

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

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Training set class distribution: {train_data[args.label_col].value_counts()}")
    print(f"Validation set class distribution: {val_data[args.label_col].value_counts()}")
    print(f"Test set class distribution: {test_data[args.label_col].value_counts()}")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    
    Loss = nn.CrossEntropyLoss()
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
            labels = labels.to(device)
            optimizer.zero_grad()
            out, decision_sfpn, decision_sfpn_t = model(imgs)
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
                labels = labels.to(device)
                out, decision_sfpn, decision_sfpn_t = model(imgs)
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
            labels = labels.to(device)
            out, decision_sfpn, decision_sfpn_t = model(imgs)
            loss = Loss(out, labels)
            test_loss += loss.item() * imgs.size(0)
            pred = torch.argmax(out, dim=1)
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