import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

# 读取 Excel 文件
df = pd.read_excel('1.xlsx', engine='openpyxl')

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))

# Cluster 曲线
plt.plot(df['Cluster_fpr'], df['Cluster_tpr'], linewidth=4, color='mediumorchid')

# Depression 曲线
plt.plot(df['Depression_fpr'], df['Depression_tpr'], linewidth=2, color='orange')

# Anxiety 曲线
plt.plot(df['Anxiety_fpr'], df['Anxiety_tpr'], linewidth=2, color='green')

# Stress 曲线
plt.plot(df['Stress_fpr'], df['Stress_tpr'], linewidth=2, color='red')

# 绘制对角线
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)

# 设置标题和其字体大小
plt.title('ROC Curves', fontsize=28, fontweight='bold')

# 设置坐标轴标签和其字体大小
plt.xlabel('False Positive Rate', fontsize=24, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=24, fontweight='bold')

# 使用自定义legend
legend_elements = [lines.Line2D([0], [0], color='mediumorchid', lw=4, label='Cluster         0.837 ± 0.088'),
                   lines.Line2D([0], [0], color='orange', lw=2, label='Depression  0.742 ± 0.066'),
                   lines.Line2D([0], [0], color='green', lw=2, label='Anxiety         0.719 ± 0.055'),
                   lines.Line2D([0], [0], color='red', lw=2, label='Stress           0.768 ± 0.079')]
# 调整图例中线的长度，并设置字体大小
plt.legend(handles=legend_elements, loc='lower right', handlelength=2, fontsize=23)

# 设置坐标轴刻度的字体大小
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig('roc.png', dpi=600, bbox_inches='tight', transparent=True)
