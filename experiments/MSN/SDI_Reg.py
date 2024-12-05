import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import argparse
import torch.optim as optim
from MSN import MSN

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
    labels = torch.tensor(labels, dtype=torch.float32)  # Change to float for regression
    return imgs, labels

def calculate_sdi_and_select(data, label_col, threshold):
    X = data.drop([label_col, 'cust_id'], axis=1).values
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    center_vector = np.mean(X_standardized, axis=0)
    distances = np.linalg.norm(X_standardized - center_vector, axis=1)
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    selected_indices = np.where(normalized_distances <= threshold)[0]
    return data.iloc[selected_indices]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSN_Arg')
    parser.add_argument('--task', type=str, default='anxiety', help='{depression, anxiety, dep_anx}')
    parser.add_argument('--task_form', type=str, default='r', help='{c for class task, r for rating task}')
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
    parser.add_argument('--csv_file', type=str, default='/home/user/xuxiao/FACES/data/baseinfo.csv')
    parser.add_argument('--root_dir', type=str, default='/home/user/xuxiao/ANRFC/data/process')
    parser.add_argument('--label_col', type=str, default='Anxiety')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (3, 96, 96, 96)
    model = MSN(input_shape=input_shape).to(device)

    # 读取CSV文件
    data = pd.read_csv(args.csv_file)
    
    data = data[data[args.label_col] != 1]
    data.loc[data[args.label_col] == 2, args.label_col] = 1

    # 定义SDI阈值范围
    sdi_thresholds = np.arange(0.9, 0.1, -0.1)
    all_results = []

    for threshold in sdi_thresholds:
        threshold = round(threshold, 2)
        print(f"\n处理SDI阈值: {threshold}")

        # 计算SDI并选择样本
        selected_data = calculate_sdi_and_select(data, args.label_col, threshold)

        # 划分训练集、验证集和测试集
        train_data, test_data = train_test_split(selected_data, test_size=0.2, random_state=42)
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

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
        
        Loss = nn.MSELoss()  # Change to MSELoss for regression
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
                out = model(imgs)
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
                    out = model(imgs)
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
        y_true = []
        y_pred = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                loss = Loss(out, labels)
                test_loss += loss.item() * imgs.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(out.cpu().numpy())
        test_loss /= len(test_dataset)

        # 计算评估指标
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        print(f'SDI Threshold: {threshold:.2f}')
        print(f'Test Loss: {test_loss:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}')

        # 保存结果
        result = {
            'SDI Threshold': threshold,
            'Test Loss': test_loss,
            'MAE': mae,
            'RMSE': rmse
        }
        all_results.append(result)

    # 将所有结果保存到CSV文件
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('msn_sdi_regression_results.csv', index=False)
    print("\n所有结果已保存到 msn_sdi_regression_results.csv")