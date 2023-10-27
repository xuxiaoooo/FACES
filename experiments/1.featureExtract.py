import os
import pandas as pd
import csv

# 获取符合条件的所有文件路径
def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and len(file) > 45:
                file_paths.append(os.path.join(root, file))
    return file_paths

def calculate_statistics(column_values):
    stats = [
        column_values.mean(),
        column_values.median(),
        column_values.max(),
        column_values.min(),
        column_values.std(),
        column_values.var(),
        column_values.max() - column_values.min(),
        column_values.quantile(0.25),
        column_values.quantile(0.75),
        column_values.quantile(0.75) - column_values.quantile(0.25),
        column_values.skew(),
        column_values.kurtosis()
    ]
    

    return stats

def process_and_append_to_csv(file_path, csv_writer, header_written):
    id = file_path.split("_")[4]
    print(id)
    df = pd.read_csv(file_path).iloc[:, 5:]
    
    stats_data = [id]
    headers = ["ID"]
    
    for column in df.columns:
        column_values = pd.to_numeric(df[column], errors='coerce').dropna()
        
        # 统计指标的名称列表
        statistics_names = ["mean", "median", "max", "min", "std", "var", "range", "q25", "q75", "iqr", "skew", "kurtosis"]
        # 计算统计指标
        stats = calculate_statistics(column_values)
        # 对每一列添加相应的统计指标名
        headers.extend([f"{column}_{stat}" for stat in statistics_names])
        
        # 添加相应列的统计值
        stats_data.extend(stats)
    
    # 如果是第一个文件，则写入 header
    if not header_written:
        csv_writer.writerow(headers)
    
    # 写入统计数据
    csv_writer.writerow(stats_data)

    return True

def main():
    directory = "/home/user/xuxiao/DATASETS/中小学面部特征"  # 当前文件夹
    file_paths = get_file_paths(directory)

    header_written = False

    with open("merged.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        for file_path in file_paths:
            header_written = process_and_append_to_csv(file_path, writer, header_written)

main()