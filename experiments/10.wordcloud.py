import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

# 读取数据
df1 = pd.read_csv('../results/features/Cluster_3.csv')
df2 = pd.read_csv('../results/features/Depression.csv')
df3 = pd.read_csv('../results/features/Anxiety.csv')
df4 = pd.read_csv('../results/features/Stress.csv')

# 提取特征和值
features1 = df1.set_index('Feature').to_dict()['Value']
features2 = df2.set_index('Feature').to_dict()['Value']
features3 = df3.set_index('Feature').to_dict()['Value']
features4 = df4.set_index('Feature').to_dict()['Value']

# 合并所有特征
all_features = {**features1, **features2, **features3, **features4}

# 对所有特征进行 Min-Max 标准化
max_value = max(all_features.values())
min_value = min(all_features.values())

normalized_features = {word: (value - min_value) / (max_value - min_value) for word, value in all_features.items()}

# 定义颜色函数
def color_func(word, **kwargs):
    if word in features1:
        return "#CD5C5C"
    elif word in features2:
        return "#4169E1"
    elif word in features3:
        return "#00FA9A"
    else:
        return "#EE82EE"

color_list = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC"]

def random_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(color_list)

wordcloud = WordCloud(
        width=2400, 
        height=300, 
        background_color='white', 
        color_func=random_color_func, 
        scale=3, 
        prefer_horizontal=1, 
        relative_scaling=1.0,
        max_words=200,          # 可以增加显示的最大词汇量
        min_font_size=10        # 可以降低最小字体大小，使得更多词可以显示
    ).generate_from_frequencies(normalized_features)

plt.figure(figsize=(24, 3))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('../draw/wordcloud.png', dpi=600, bbox_inches='tight')
