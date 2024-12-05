import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import argparse
from LI_FPN import LI_FPN
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    def __init__(self, data, root_dir, label_col, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.label_col = label_col
        self.transform = transform
        self.valid_ids = self._get_valid_ids()
        print(f"Total valid samples: {len(self.valid_ids)}")

    def _get_valid_ids(self):
        valid_ids = []
        for idx in range(len(self.data)):
            img_id = self.data.iloc[idx]['cust_id']
            img_folder = os.path.join(self.root_dir, str(img_id))
            if os.path.exists(img_folder) and all(os.path.exists(os.path.join(img_folder, f'{i}.jpg')) for i in range(1, 14, 2)):
                valid_ids.append(idx)
        return valid_ids

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        data_idx = self.valid_ids[idx]
        img_id = self.data.iloc[data_idx]['cust_id']
        label = self.data.iloc[data_idx][self.label_col]
        img_folder = os.path.join(self.root_dir, str(img_id))
        imgs = []
        for i in range(1, 14, 2):
            img_path = os.path.join(img_folder, f'{i}.jpg')
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
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
        
        num_samples_to_remove = int(len(score_indices) * threshold)
        
        if num_samples_to_remove < len(score_indices):
            indices_to_keep = score_indices[np.argsort(score_distances)[:-num_samples_to_remove]]
        else:
            indices_to_keep = [score_indices[np.argmin(score_distances)]]
        
        selected_indices.extend(indices_to_keep)
    
    return data.iloc[selected_indices]

def get_binary_labels_for_dass_factor(df, factor_name):
    if factor_name == 'Depression':
        return (df['Depression'] > 9).astype(int)
    elif factor_name == 'Anxiety':
        return (df['Anxiety'] > 7).astype(int)
    elif factor_name == 'Stress':
        return (df['Stress'] > 14).astype(int)
    else:
        raise ValueError("Unknown factor name!")

def balance_data(data, label_col):
    class_0 = data[data[label_col] == 0]
    class_1 = data[data[label_col] == 1]

    n_samples = min(len(class_0), len(class_1))

    class_0_downsampled = resample(class_0, replace=False, n_samples=n_samples, random_state=42)
    class_1_downsampled = resample(class_1, replace=False, n_samples=n_samples, random_state=42)

    balanced_data = pd.concat([class_0_downsampled, class_1_downsampled]).reset_index(drop=True)
    return balanced_data

def main():
    parser = argparse.ArgumentParser('LI_FPN_Arg')
    parser.add_argument('--task', type=str, default='anxiety', help='{depression, anxiety, stress}')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--task_form', type=str, default='c', help='{c for class task, r for rating task}')
    parser.add_argument('--lim', type=str, default='g', help='{g for LI_G, s for LI_S}')
    parser.add_argument('--backbone', type=str, default='res18', help='{res18, res34, res50}')
    parser.add_argument('--len_t', type=int, default=50, help='the length of input')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--log_path', type=str, default='log path, a txt file')
    parser.add_argument('--dass21_file', type=str, default='/home/user/xuxiao/FACES/data/dass21.csv')
    parser.add_argument('--root_dir', type=str, default='/home/user/xuxiao/FACES/data/process')
    args = parser.parse_args()

    # 读取DASS21数据
    dass21_scores = pd.read_csv(args.dass21_file)

    # 计算DASS21得分
    dass21_scores['Depression'] = dass21_scores[['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21']].sum(axis=1)
    dass21_scores['Anxiety'] = dass21_scores[['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20']].sum(axis=1)
    dass21_scores['Stress'] = dass21_scores[['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']].sum(axis=1)

    dass_columns = {
        'Depression': ['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21'],
        'Anxiety': ['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20'],
        'Stress': ['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']
    }

    print(f"Original data shape: {dass21_scores.shape}")

    factors = ['Depression', 'Anxiety', 'Stress']
    results = []

    for factor in factors:
        print(f"\n处理因子: {factor}")
        args.task = factor.lower()

        for threshold in np.arange(0.1, 0.5, 0.1):
            print(f"\n处理 SDI 阈值: {threshold}")
            
            # 应用SDI筛选
            selected_data = calculate_sdi_and_select(dass21_scores, dass_columns[factor], factor, threshold)
            print(f"SDI 后的数据形状: {selected_data.shape}")

            # 获取二元标签
            selected_data['label'] = get_binary_labels_for_dass_factor(selected_data, factor)
            print(f"标记后的类别分布: {selected_data['label'].value_counts()}")

            # 平衡数据
            balanced_data = balance_data(selected_data, 'label')
            print(f"平衡后的数据形状: {balanced_data.shape}")
            print(f"平衡后的类别分布: {balanced_data['label'].value_counts()}")

            # 划分训练集、验证集和测试集
            train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=42, stratify=balanced_data['label'])
            val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42, stratify=test_data['label'])

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = ImageDataset(train_data, args.root_dir, 'label', transform)
            val_dataset = ImageDataset(val_data, args.root_dir, 'label', transform)
            test_dataset = ImageDataset(test_data, args.root_dir, 'label', transform)

            print(f"Number of training samples: {len(train_dataset)}")
            print(f"Number of validation samples: {len(val_dataset)}")
            print(f"Number of test samples: {len(test_dataset)}")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LI_FPN(class_num=args.class_num, task_form=args.task_form, lim=args.lim, 
                           backbone=args.backbone, len_t=args.len_t, pretrain=args.pretrain).to(device)
            Loss = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            patience = 50
            best_val_loss = float('inf')
            counter = 0

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

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)

            conf_matrix = confusion_matrix(y_true, y_pred)
            print(f'Confusion Matrix:\n{conf_matrix}')

            print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}')
            print(f'Accuracy: {acc:.3f}, F1-score: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}')

            results.append({
                'Factor': factor,
                'Threshold': threshold,
                'Accuracy': acc,
                'F1-score': f1,
                'Recall': recall,
                'Precision': precision
            })

    # 输出结果表格
    results_df = pd.DataFrame(results)
    print("\n结果:")
    print(results_df.to_string(index=False))

    # 保存结果到CSV文件
    results_df.to_csv('sdi_classification_results.csv', index=False)
    print("\n结果已保存到 sdi_classification_results.csv")

if __name__ == '__main__':
    main()
