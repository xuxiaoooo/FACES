import pandas as pd

def label_2():
    # 读取数据
    df = pd.read_csv('/home/user/xuxiao/miniFacial/data/clusters.csv')
    score = pd.read_csv('/home/user/xuxiao/miniFacial/data/baseinfo.csv')

    # 确保两个数据集都按 cust_id 排序
    df = df.sort_values('cust_id')
    score = score.sort_values('cust_id')

    # 使用 DASS-21 的分数为每一列进行划分
    score['Depression_class'] = (score['Depression'] > 9).astype(int)
    score['Anxiety_class'] = (score['Anxiety'] > 7).astype(int)
    score['Stress_class'] = (score['Stress'] > 14).astype(int)

    # 从 score 中选择新列并添加到 df 中
    df = pd.merge(df, score[['cust_id', 'Depression_class', 'Anxiety_class', 'Stress_class']], on='cust_id', how='left')
    print(df['Depression_class'].value_counts())
    # 如果需要，可以保存到新的 CSV 文件
    df.to_csv('/home/user/xuxiao/miniFacial/data/label.csv', index=False)

def label_3():
    # 读取数据
    df = pd.read_csv('/home/user/xuxiao/miniFacial/data/clusters.csv')
    score = pd.read_csv('/home/user/xuxiao/miniFacial/data/baseinfo.csv')

    # 确保两个数据集都按 cust_id 排序
    df = df.sort_values('cust_id')
    score = score.sort_values('cust_id')

    # 使用 DASS-21 的分数进行三类划分
    # 对于抑郁症
    score['Depression_class'] = 0  # 默认为 0（无）
    score.loc[score['Depression'] > 9, 'Depression_class'] = 1  # 轻度
    score.loc[score['Depression'] > 13, 'Depression_class'] = 2  # 轻度以上

    # 对于焦虑症
    score['Anxiety_class'] = 0  # 默认为 0（无）
    score.loc[score['Anxiety'] > 7, 'Anxiety_class'] = 1  # 轻度
    score.loc[score['Anxiety'] > 9, 'Anxiety_class'] = 2  # 轻度以上

    # 对于压力
    score['Stress_class'] = 0  # 默认为 0（无）
    score.loc[score['Stress'] > 14, 'Stress_class'] = 1  # 轻度
    score.loc[score['Stress'] > 18, 'Stress_class'] = 2  # 轻度以上

    # 从 score 中选择新列并添加到 df 中
    df = pd.merge(df, score[['cust_id', 'Depression_class', 'Anxiety_class', 'Stress_class']], on='cust_id', how='left')

    print(df['Depression_class'].value_counts())
    print(df['Anxiety_class'].value_counts())
    print(df['Stress_class'].value_counts())

    # 如果需要，可以保存到新的 CSV 文件
    df.to_csv('/home/user/xuxiao/miniFacial/data/label_3.csv', index=False)

# 调用函数
label_3()