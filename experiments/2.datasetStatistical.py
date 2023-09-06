import pandas as pd
from datetime import datetime
from statistics import mean, stdev


# 读取第一个CSV文件
df = pd.read_csv("/home/user/xuxiao/miniFacial/data/baseinfo.csv")

# 计算年龄分布信息
df['birthday'] = pd.to_datetime(df['birthday'], format='%Y%m%d')
df['age'] = (datetime(2022, 9, 1) - df['birthday']).dt.days // 365
mean_age = df['age'].mean()
sd_age = df['age'].std()

print(f"Average Age: {mean_age:.3f}")
print(f"Age Standard Deviation: {sd_age:.3f}")
print("\nAge Distribution:")
print(df['age'].value_counts())

# 统计每个 grade_name 中 gender 的数量分布，并增加 Total 列
gender_distribution = df.groupby(['grade_name', 'gender']).size().unstack().fillna(0)
gender_distribution['Total'] = gender_distribution.sum(axis=1)
print("\nGender Distribution in each Grade with Total count:")
print(gender_distribution)

# 使用 DASS-21 标准分析 depression, anxiety 和 pressure 的分布
DASS_21 = {
    'depression': {'normal': (0, 9), 'mild': (10, 13), 'moderate': (14, 20), 'severe': (21, 27), 'extremely severe': (28, 42)},
    'anxiety': {'normal': (0, 7), 'mild': (8, 9), 'moderate': (10, 14), 'severe': (15, 19), 'extremely severe': (20, 42)},
    'pressure': {'normal': (0, 14), 'mild': (15, 18), 'moderate': (19, 25), 'severe': (26, 33), 'extremely severe': (34, 42)}
}

for column, scales in DASS_21.items():
    df[f'{column}_level'] = pd.cut(df[column], 
                                  bins=[-1] + [v[1] for k, v in scales.items()], 
                                  labels=list(scales.keys()))
    print(f"\n{column.capitalize()} Distribution based on DASS-21:")
    print(df[f'{column}_level'].value_counts())

# 输出总人数
total_count = df['cust_id'].nunique()
print(f"\nTotal Number of Students: {total_count}")