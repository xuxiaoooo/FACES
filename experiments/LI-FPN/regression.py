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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import matplotlib.pyplot as plt
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
        return imgs, label

def custom_collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.float32)  # 将标签类型改为float32
    return imgs, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LI_FPN_Arg')
    parser.add_argument('--task', type=str, default='depression', help='{depression, anxiety, dep_anx}')
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--task_form', type=str, default='r', help='{c for class task, r for rating task}')  # 将任务形式改为回归
    parser.add_argument('--lim', type=str, default='g', help='{g for LI_G, s for LI_S}')
    parser.add_argument('--backbone', type=str, default='res18', help='{res18, res34, res50}')
    parser.add_argument('--len_t', type=int, default=7, help='the length of input')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--model_id', type=str, default='define your model name')
    parser.add_argument('--log_path', type=str, default='log path, a txt file')
    parser.add_argument('--model_save_path', type=str, default='where the trained model is saved, a folder path')
    parser.add_argument('--csv_file', type=str, default='/home/user/xuxiao/FACES/data/baseinfo.csv')
    parser.add_argument('--root_dir', type=str, default='/home/user/xuxiao/FACES/data/process')
    parser.add_argument('--label_col', type=str, default='Stress')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LI_FPN(task_form=args.task_form,
                   class_num=args.class_num,
                   lim=args.lim,
                   backbone=args.backbone,
                   len_t=args.len_t,
                   pretrain=args.pretrain).to(device)

    # 读取CSV文件
    data = pd.read_csv(args.csv_file)

    # 划分训练集、验证集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=1/2, random_state=42)

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    Loss = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    patience_counter = 0

    # 训练循环
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out, decision_sfpn, decision_sfpn_t = model(imgs)
            loss = Loss(out.squeeze(), labels)  # 将输出压缩为一维向量
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
                loss = Loss(out.squeeze(), labels)  # 将输出压缩为一维向量
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_dataset)

        print(f'Epoch [{epoch+1}/{args.epoch}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 测试模型
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out, decision_sfpn, decision_sfpn_t = model(imgs)
            loss = Loss(out.squeeze(), labels)  # 将输出压缩为一维向量
            test_loss += loss.item() * imgs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(out.squeeze().cpu().numpy())
    test_loss /= len(test_dataset)

    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f'Test Loss: {test_loss:.3f}')
    print(f'Mean Absolute Error (MAE): {mae:.3f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')

    # 绘制真实值和预测值的散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='gray')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='lightblue', linestyle='--', label='y=x')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    # plt.title('True vs Predicted Values')
    plt.legend()
    plt.savefig('scatter_plot.jpg', dpi=300, bbox_inches='tight')