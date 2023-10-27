import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../data/baseinfo.csv")

factors = ['Depression', 'Anxiety', 'Stress']

# 描述性统计
print(data[factors].describe())

# 相关性分析
correlation_matrix = data[factors].corr()
print(correlation_matrix)

# 可视化
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.savefig('../draw/correlation_dass21.png', dpi=600, bbox_inches='tight', transparent=True)

for dependent in factors:
    independents = [f for f in factors if f != dependent]
    X = data[independents]
    X = sm.add_constant(X)
    y = data[dependent]
    
    model = sm.OLS(y, X).fit()
    print(f"Regression Analysis with {dependent} as dependent variable:\n")
    print(model.summary())
    print("\n\n")

