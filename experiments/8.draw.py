import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd


# silhouette
# values = [0.578, 0.514, 2 - 0.578 - 0.514]
# colors = ['#E5C494', '#8DA0CB', 'white']

# fig, ax = plt.subplots()
# ax.pie(values, labels=None, colors=colors, startangle=90, wedgeprops=dict(width=0.2,edgecolor='black'))

# plt.savefig('../draw/silhouette.png', dpi=600, bbox_inches='tight', transparent=True)

# distribution Cluster pie 2
# values = [8185, 2876]
# colors = ['#8DA0CB', '#F58D63']

# fig, ax = plt.subplots()
# ax.pie(values, labels=None, colors=colors, startangle=90, wedgeprops=dict(width=1,edgecolor='black'))

# plt.savefig('../draw/cluster2pie.png', dpi=600, bbox_inches='tight', transparent=True)

# distribution Cluster pie 3
# values = [6714, 3381, 965]
# colors = ['#8DA0CB', '#F58D63', '#66C2A5']

# fig, ax = plt.subplots()
# ax.pie(values, labels=None, colors=colors, startangle=90, wedgeprops=dict(width=1,edgecolor='black'))

# plt.savefig('../draw/cluster3pie.png', dpi=600, bbox_inches='tight', transparent=True)


# U 检验
# data = [(' eye_lmk_Y', 554.3116895326209), (' Y', 459.4839023921774), (' y', 364.5196683304533), (' eye_lmk_y', 309.07798687972763), (' x', 123.46786442424873), (' eye_lmk_x', 122.03148908130835), (' p', 121.5584590214175), (' eye_lmk_Z', 89.93385735410055), (' X', 70.10221421345497), (' eye_lmk_X', 67.43226445743066), (' Z', 53.504899626529365), (' AU02', 14.36924787678907), (' pose', 13.537665911412684), (' AU20', 12.838186782660188), (' AU17', 12.239783481272816), (' AU25', 10.217777785727408), (' gaze_0', 10.02313350747147), (' AU23', 10.016473943966231), (' AU09', 9.972824196661206), (' AU01', 9.84305518846184), (' AU05', 9.603793012729302), (' AU04', 8.229642118129588), (' AU15', 8.140894769257697), (' AU26', 7.940525643573976), (' AU45', 7.5068631773902394), (' gaze_1', 6.947398757601312), (' AU28', 6.302460394564359), (' AU14', 6.1298772368469185), (' AU07', 4.719418885163049), (' gaze_angle', 3.0524113693940897), (' AU06', 2.994726342886667), (' AU12', 2.5012904933011613), (' AU10', 0.7221982341011102)]
# plt.rc('font', family='Arial')
# plt.rc('font', size=12)             # 设置全局默认字体大小
# plt.rc('axes', labelsize=14)        # 设置轴标签字体大小
# plt.rc('xtick', labelsize=12)       # 设置x轴刻度标签字体大小
# plt.rc('ytick', labelsize=12) 
# sns.set_theme(style="whitegrid")
# df = pd.DataFrame(data[:20], columns=["abbrev", "value"])
# df = df.sort_values(by="value", ascending=False)
# g = sns.PairGrid(df,
#                 x_vars=["value"], y_vars=["abbrev"],
#                 height=6, aspect=.25)
# g.map(sns.stripplot, size=10, orient="h", jitter=False,
#     palette="flare_r", linewidth=0.6, edgecolor="w")
# g.set(xlim=(-100, df["value"].max() * 1.2), xlabel="p-value", ylabel="")
# for ax in g.axes.flat:
#     ax.xaxis.grid(False)
#     ax.yaxis.grid(True)
# sns.despine(left=True, bottom=True)
# plt.savefig('../draw/utest.png', dpi=600, bbox_inches='tight', transparent=True)