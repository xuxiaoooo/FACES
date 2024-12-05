import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('../data/dass21.csv')

# 提取DASS-21量表的列
dass_columns = [f'dass_{i}' for i in range(1, 22)]

# 提取子分量表对应的列
depression_columns = ['dass_3', 'dass_5', 'dass_10', 'dass_13', 'dass_16', 'dass_17', 'dass_21']
anxiety_columns = ['dass_2', 'dass_4', 'dass_7', 'dass_9', 'dass_15', 'dass_19', 'dass_20']
stress_columns = ['dass_1', 'dass_6', 'dass_8', 'dass_11', 'dass_12', 'dass_14', 'dass_18']

# 标准化处理
def standardize(df, columns):
    return (df[columns] - df[columns].mean()) / df[columns].std()

df_standardized = df.copy()
df_standardized[dass_columns] = standardize(df, dass_columns)

# 将抑郁、焦虑和压力程度分类
def categorize_severity(df):
    # Depression
    depression_bins = [-1, 9, 13, 20, 27, float('inf')]
    labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    df['Depression_Severity'] = pd.cut(df['Depression'], bins=depression_bins, labels=labels)

    # Anxiety
    anxiety_bins = [-1, 7, 9, 14, 19, float('inf')]
    df['Anxiety_Severity'] = pd.cut(df['Anxiety'], bins=anxiety_bins, labels=labels)

    # Stress
    stress_bins = [-1, 14, 18, 25, 33, float('inf')]
    df['Stress_Severity'] = pd.cut(df['Stress'], bins=stress_bins, labels=labels)

# 对每个维度进行分类
categorize_severity(df_standardized)

# 绘图函数，修改以绘制多个百分比的数据在同一张图上
def plot_chart_multiple_percentages(data_dict, title, x_labels, file_name, is_severity=False):
    if not is_severity:
        fig_size = (34, 6)
        legend_fontsize = 25
        tick_labelsize = 25
    else:
        fig_size = (13, 6)
        legend_fontsize = 25
        tick_labelsize = 25

    # 设置全局图例字体大小
    plt.rcParams['legend.fontsize'] = legend_fontsize

    fig, ax = plt.subplots(figsize=fig_size)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    percentages = data_dict.keys()
    colors = ['#F5D0E1', '#BCDE79', '#CBCBCB', '#BED3E3']
    shifts = np.linspace(-0.3, 0.3, len(percentages))

    for idx, (percent, data) in enumerate(data_dict.items()):
        shift = shifts[idx]
        positions = np.arange(1, len(data['Mean']) + 1) + shift

        violin_data = []
        for i in range(len(data['Mean'])):
            violin_data.append(np.random.uniform(data['Min'][i], data['Max'][i], 100))

        # Plot the violin plot
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            widths=0.16,
            showmeans=False,
            showmedians=False
        )
        for pc in parts['bodies']:
            pc.set_facecolor(colors[idx])
            pc.set_edgecolor('none')
            pc.set_alpha(1)

        # Plot the mean line
        ax.plot(
            positions,
            data['Mean'],
            color=colors[idx],
            marker='o',
            linewidth=5,
            markersize=5,
            label=f'Treshold:{percent/100}'
        )

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels, fontname='Arial', fontweight='bold', fontsize=tick_labelsize)

    # Remove titles and axis labels
    # ax.set_ylabel('SDI Value', fontsize=15, fontweight='bold', fontname='Arial')
    # ax.set_title(title, fontsize=20, fontweight='bold', fontname='Arial')

    # Remove the top x-axis labels (if any)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', which='both', bottom=True, top=False)

    # Set y-axis tick parameters to match the font
    ax.tick_params(axis='y', labelsize=tick_labelsize, width=4)
    plt.setp(ax.get_yticklabels(), fontname='Arial', fontweight='bold')

    # 设置 x 轴范围，减少前后空白
    ax.set_xlim(0.5, len(x_labels) + 0.5)

    # 修改图例设置
    if not is_severity:
        legend = ax.legend(
            frameon=False,
            loc='best',
            prop={'family': 'Arial', 'weight': 'bold', 'size': legend_fontsize}
        )

        # 手动设置图例中每个元素的字体大小
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
    else:
        ax.legend().remove()  # 如果是 severity 图表，则移除图例

    # 调整布局并保存图形
    plt.tight_layout()
    if is_severity:
        plt.subplots_adjust(right=0.85)
    plt.savefig('/home/user/xuxiao/FACES/draw/SDI-Treshold/'+file_name, dpi=800, format='jpg', bbox_inches='tight')
    plt.close()

    # 重置全局设置，避免影响其他图表
    plt.rcParams['legend.fontsize'] = plt.rcParamsDefault['legend.fontsize']

# 计算中心向量
def calculate_center_vector(df, group_column, target_columns):
    groups = df.groupby(group_column)
    center_vectors = {}
    for group_name, group in groups:
        center_vector = group[target_columns].mean()
        center_vectors[group_name] = center_vector
    return center_vectors

# 计算个体到中心向量的距离
def calculate_distances(df, group_column, target_columns, center_vectors):
    distances = []
    for index, row in df.iterrows():
        group = row[group_column]
        center_vector = center_vectors[group]
        distance = np.sqrt(((row[target_columns] - center_vector) ** 2).sum())
        distances.append(distance)
    return distances

# 归一化处理
def normalize_distances(distances):
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    return normalized_distances

# 计算均值距离和标准差距离
def calculate_g_sddi(df, group_column, target_columns):
    center_vectors = calculate_center_vector(df, group_column, target_columns)
    distances = calculate_distances(df, group_column, target_columns, center_vectors)
    normalized_distances = normalize_distances(distances)
    df['distance'] = normalized_distances
    mean_distance = df.groupby(group_column)['distance'].mean()
    std_distance = df.groupby(group_column)['distance'].std()
    summary = df.groupby(group_column)['distance'].agg(['mean', 'std', 'min', 'max', 'count'])
    summary = summary.reset_index()
    return mean_distance, std_distance, summary, df

# 计算加权平均差异指数（SDI）
def calculate_wadi(mean_distance, std_distance, counts):
    weighted_sum = 0
    total_count = 0
    for group in mean_distance.index:
        if group in counts.index:
            n = counts[group]
            md = mean_distance[group]
            sd = std_distance[group]
            weighted_sum += n * (md + sd)
            total_count += n
    if total_count == 0:
        return 0  # 避免除以零的错误
    wadi = weighted_sum / total_count
    return wadi

# 定义需要移除的百分比
percentages = [0, 10, 20, 30]

# 定义维度和对应的列
dimensions = {
    'Depression': depression_columns,
    'Anxiety': anxiety_columns,
    'Stress': stress_columns
}

severity_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']

# 存储绘图数据
plot_data = {}

# 循环处理每个维度
for dimension, columns in dimensions.items():
    plot_data[dimension] = {'Same Score': {}, 'Same Severity': {}}
    print(f"\nProcessing for {dimension}...")

    # 处理 Same Score
    group_column = dimension

    # 存储各个百分比的数据
    data_dict = {}

    for percent in percentages:
        print(f"  Calculating for top {percent}% distances removed...")

        df_filtered = df_standardized.copy()

        # 计算均值距离和标准差距离
        mean_distance, std_distance, summary, df_with_distance = calculate_g_sddi(df_filtered, group_column, columns)

        # 根据百分比移除数据
        if percent > 0:
            def remove_top_percent(df, group_column, percent):
                df_removed = pd.DataFrame()
                for group in df[group_column].unique():
                    group_df = df[df[group_column] == group]
                    threshold = np.percentile(group_df['distance'], 100 - percent)
                    group_df = group_df[group_df['distance'] <= threshold]
                    df_removed = pd.concat([df_removed, group_df])
                return df_removed

            df_filtered = remove_top_percent(df_with_distance, group_column, percent)
            # ��新计算
            mean_distance, std_distance, summary, _ = calculate_g_sddi(df_filtered, group_column, columns)

        # 计算加权平均差异指数（SDI）
        wadi = calculate_wadi(mean_distance, std_distance, summary['count'])

        # 准备绘图数据
        data = {
            'Mean': mean_distance.tolist(),
            'STD': std_distance.tolist(),
            'Min': summary['min'].tolist(),
            'Max': summary['max'].tolist(),
            'Count': summary['count'].astype(int).tolist(),
            'Group': summary[group_column].tolist()
        }

        data_dict[percent] = data

    # 准备 x_labels
    x_labels = data_dict[0]['Group']

    # 绘制图形
    title = f'{dimension} SDI (Same Score)'
    filename = f'{dimension.lower()}_same_score_combined.jpg'
    plot_chart_multiple_percentages(data_dict, title, x_labels, filename, is_severity=False)

    # 处理 Same Severity
    severity_column = f'{dimension}_Severity'

    # 存储各个百分比的数据
    data_dict_severity = {}

    for percent in percentages:
        print(f"  Calculating for top {percent}% distances removed (Severity)...")

        df_filtered = df_standardized.copy()

        # 计算均值距离和标准差距离
        mean_distance, std_distance, summary, df_with_distance = calculate_g_sddi(df_filtered, severity_column, columns)

        # 根据百分比移除数据
        if percent > 0:
            df_filtered = remove_top_percent(df_with_distance, severity_column, percent)
            # 重新计算
            mean_distance, std_distance, summary, _ = calculate_g_sddi(df_filtered, severity_column, columns)

        # 计算加权平均差异指数（SDI）
        wadi = calculate_wadi(mean_distance, std_distance, summary['count'])

        # 准备绘图数据
        data = {
            'Mean': mean_distance.tolist(),
            'STD': std_distance.tolist(),
            'Min': summary['min'].tolist(),
            'Max': summary['max'].tolist(),
            'Count': summary['count'].astype(int).tolist(),
            'Group': summary[severity_column].tolist()
        }

        data_dict_severity[percent] = data

    # 准备 x_labels
    x_labels_severity = severity_labels

    # 绘制图形
    title = f'{dimension} SDI (Same Severity)'
    filename = f'{dimension.lower()}_same_severity_combined.jpg'
    plot_chart_multiple_percentages(data_dict_severity, title, x_labels_severity, filename, is_severity=True)

print("\nEvaluation Criteria:")
print("Value range [0, 1]. SDI value close to 0 indicates small discrepancy in subscale distribution; SDI value close to 1 indicates large discrepancy in subscale distribution.")


# import matplotlib.pyplot as plt
# import numpy as np

# # Depression SDI (Same Score)
# depression_same_score = {
#     'Mean': [0.0000, 0.1616, 0.2458, 0.3034, 0.3315, 0.3514, 0.3500, 0.3157, 0.3759, 0.4480, 0.4718, 0.4711, 0.4787, 0.5059, 0.4764, 0.5039, 0.4655, 0.4358, 0.3864, 0.3086, 0.1951, 0.0000],
#     'STD': [0.0000, 0.0647, 0.0657, 0.1060, 0.1025, 0.1189, 0.1521, 0.1884, 0.1625, 0.1752, 0.1648, 0.1552, 0.1560, 0.1564, 0.1598, 0.1499, 0.1318, 0.1223, 0.0958, 0.0814, 0.0271, 0.0000],
#     'Min': [0.0000, 0.1028, 0.1790, 0.1803, 0.1983, 0.1875, 0.1573, 0.1062, 0.1409, 0.1747, 0.1901, 0.2017, 0.1840, 0.2177, 0.1814, 0.1693, 0.2327, 0.2209, 0.2358, 0.2117, 0.1701, 0.0000],
#     'Max': [0.0000, 0.2869, 0.5562, 0.8252, 0.7407, 0.8335, 1.0000, 0.8548, 0.9196, 0.9398, 0.9693, 0.8405, 0.9557, 0.8621, 0.7798, 0.9340, 0.8234, 0.6844, 0.5225, 0.5486, 0.2707, 0.0000],
#     'Count': [4169, 1732, 1170, 1000, 725, 530, 456, 445, 268, 204, 148, 117, 91, 87, 68, 54, 41, 32, 26, 32, 14, 18]
# }

# # Anxiety SDI (Same Score)
# anxiety_same_score = {
#     'Mean': [0.0000, 0.1482, 0.2185, 0.2985, 0.3376, 0.3726, 0.3907, 0.3664, 0.4207, 0.4528, 0.4656, 0.4699, 0.5127, 0.4923, 0.5016, 0.5240, 0.4621, 0.4790, 0.4697, 0.3222, 0.1869, 0.0000],
#     'STD': [0.0000, 0.0627, 0.0896, 0.0822, 0.0867, 0.0917, 0.1334, 0.1721, 0.1834, 0.1776, 0.1557, 0.1626, 0.1813, 0.1494, 0.1814, 0.1439, 0.0801, 0.1224, 0.0862, 0.0718, 0.0443, 0.0000],
#     'Min': [0.0000, 0.0973, 0.1141, 0.2115, 0.2495, 0.2729, 0.2544, 0.1548, 0.1034, 0.1213, 0.2044, 0.2186, 0.2359, 0.2102, 0.1301, 0.1297, 0.2467, 0.2409, 0.3242, 0.2164, 0.1497, 0.0000],
#     'Max': [0.0000, 0.2866, 0.5504, 0.7147, 0.8082, 0.8483, 0.9570, 0.9001, 0.9867, 1.0000, 0.9738, 0.9669, 0.9101, 0.8599, 0.8255, 0.9014, 0.6347, 0.6698, 0.6560, 0.4941, 0.2376, 0.0000],
#     'Count': [3327, 1764, 1328, 1156, 848, 643, 535, 500, 353, 251, 180, 141, 113, 75, 56, 57, 31, 24, 19, 10, 4, 12]
# }

# # Stress SDI (Same Score)
# stress_same_score = {
#     'Mean': [0.0000, 0.2151, 0.2983, 0.3557, 0.3658, 0.3716, 0.3601, 0.2985, 0.4011, 0.4500, 0.4692, 0.4884, 0.5220, 0.5130, 0.4573, 0.4933, 0.4494, 0.4209, 0.4425, 0.3135, 0.2159, 0.0000],
#     'STD': [0.0000, 0.0211, 0.0529, 0.1009, 0.0984, 0.1195, 0.1425, 0.2202, 0.1716, 0.1618, 0.1570, 0.1697, 0.1709, 0.1534, 0.1422, 0.1390, 0.1243, 0.1066, 0.1254, 0.1184, 0.0140, 0.0000],
#     'Min': [0.0000, 0.1918, 0.2369, 0.2330, 0.2657, 0.2581, 0.2073, 0.0532, 0.1639, 0.2131, 0.2164, 0.2282, 0.1922, 0.1207, 0.2257, 0.2230, 0.2336, 0.2239, 0.2191, 0.2147, 0.2054, 0.0000],
#     'Max': [0.0000, 0.2587, 0.4997, 0.7534, 0.7391, 0.8021, 0.9017, 0.8660, 0.8750, 0.9465, 0.9589, 0.8982, 1.0000, 0.8221, 0.9069, 0.8773, 0.7872, 0.6644, 0.5586, 0.5271, 0.2451, 0.0000],
#     'Count': [2964, 1249, 1018, 1010, 936, 810, 778, 743, 522, 343, 270, 189, 186, 123, 71, 58, 51, 38, 29, 20, 8, 11]
# }

# # Depression SDI (Same Severity)
# depression_same_severity = {
#     'Severity': ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'],
#     'Mean': [0.0972, 0.2736, 0.3150, 0.4250, 0.4067],
#     'STD': [0.1287, 0.1479, 0.2029, 0.1719, 0.1489],
#     'Min': [0.0000, 0.0651, 0.0292, 0.0905, 0.1186],
#     'Max': [0.7612, 1.0000, 0.9968, 0.9241, 0.9407],
#     'Count': [8796, 986, 1065, 295, 285]
# }

# # Anxiety SDI (Same Severity)
# anxiety_same_severity = {
#     'Severity': ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'],
#     'Mean': [0.1309, 0.3215, 0.3642, 0.4442, 0.4612],
#     'STD': [0.1190, 0.1201, 0.1916, 0.1788, 0.1376],
#     'Min': [0.0331, 0.1978, 0.0000, 0.1550, 0.0798],
#     'Max': [0.8040, 0.9546, 1.0000, 0.9131, 0.8879],
#     'Count': [8423, 1178, 1284, 329, 213]
# }

# # Stress SDI (Same Severity)
# stress_same_severity = {
#     'Severity': ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'],
#     'Mean': [0.1507, 0.2883, 0.3216, 0.4509, 0.4144],
#     'STD': [0.1445, 0.1476, 0.2138, 0.1878, 0.1365],
#     'Min': [0.0197, 0.1092, 0.0000, 0.0621, 0.1565],
#     'Max': [0.7445, 0.8896, 0.9200, 1.0000, 0.8831],
#     'Count': [7177, 1588, 1878, 498, 286]
# }

# def plot_chart(data, title, x_labels, file_name, is_severity=False):
#     fig, ax = plt.subplots(figsize=(50, 10) if not is_severity else (8, 3))

#     # Create violin plot data
#     violin_data = []
#     for i in range(len(data['Mean'])):
#         violin_data.append(np.random.uniform(data['Min'][i], data['Max'][i], 100))

#     # Plot the violin plot
#     parts = ax.violinplot(violin_data, showmeans=False, showmedians=False)
#     for pc in parts['bodies']:
#         pc.set_facecolor('#2e6f9a')
#         pc.set_edgecolor('#366381')
#         pc.set_alpha(0.6)

#     # Plot the mean line
#     ax.plot(range(1, len(data['Mean']) + 1), data['Mean'], color='#b53e3b', marker='o', linewidth=8 if not is_severity else 2, markersize=8, label='Mean')

#     # Set x-axis ticks and labels
#     ax.set_xticks(range(1, len(x_labels) + 1))
#     ax.set_xticklabels(x_labels, fontsize=60 if not is_severity else 15, fontweight='bold', fontname='Arial')

#     # Set y-axis label
#     ax.set_ylabel('', fontsize=60 if not is_severity else 15, fontweight='bold', fontname='Arial')

#     # Set y-axis tick parameters to match the font
#     ax.tick_params(axis='y', labelsize=60 if not is_severity else 15, width=2)

#     # Add sample size information
#     for i, count in enumerate(data['Count']):
#         ax.text(i + 1, ax.get_ylim()[1], str(count), ha='center', va='bottom', fontsize=60 if not is_severity else 15, fontweight='bold', fontname='Arial')

#     # Add legend
#     ax.legend(fontsize=60 if not is_severity else 15, frameon=False)

#     # Adjust layout and save the figure
#     plt.tight_layout()
#     plt.savefig(file_name, dpi=600, format='jpg')
#     plt.close()

# # Plot and save the figures
# plot_chart(depression_same_score, 'Depression SDI (Same Score)', range(0, 42, 2), 'depression_same_score.jpg')
# plot_chart(anxiety_same_score, 'Anxiety SDI (Same Score)', range(0, 42, 2), 'anxiety_same_score.jpg')
# plot_chart(stress_same_score, 'Stress SDI (Same Score)', range(0, 42, 2), 'stress_same_score.jpg')

# plot_chart(depression_same_severity, 'Depression SDI (Same Severity)', depression_same_severity['Severity'], 'depression_same_severity.jpg', True)
# plot_chart(anxiety_same_severity, 'Anxiety SDI (Same Severity)', anxiety_same_severity['Severity'], 'anxiety_same_severity.jpg', True)
# plot_chart(stress_same_severity, 'Stress SDI (Same Severity)', stress_same_severity['Severity'], 'stress_same_severity.jpg', True)

