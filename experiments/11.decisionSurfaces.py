import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def load_data(scale, label_path, feature_path):
    df_label = pd.read_csv(label_path)[['cust_id', scale]]
    df_label = df_label.loc[df_label[scale] != 1]
    df_label[scale].replace(2, 1, inplace=True)
    df_features = pd.read_csv(feature_path)
    merged_df = pd.merge(df_features, df_label, left_on='ID', right_on='cust_id', how='inner')
    X = merged_df.drop(columns=['cust_id', 'ID', scale])
    y = merged_df[scale]
    return X, y

def random_down_sampling(X, y, random_state):
    df = pd.concat([X, y], axis=1)
    # 分离多数和少数类
    class_counts = df[y.name].value_counts()
    majority_class_label = class_counts.idxmax()
    minority_class_label = class_counts.idxmin()
    majority_class = df[df[y.name] == majority_class_label]
    minority_class = df[df[y.name] == minority_class_label]
    # 下采样多数类
    majority_downsampled = resample(majority_class, 
                                    replace=False,  # 不放回抽样
                                    n_samples=len(minority_class),  # 将多数类样本数减少到与少数类一样
                                    random_state=random_state)  # 可重现的结果
    # 合并少数类和下采样后的多数类
    downsampled_df = pd.concat([majority_downsampled, minority_class])
    # 分离特征和标签
    X_downsampled = downsampled_df.drop(y.name, axis=1).reset_index(drop=True)
    y_downsampled = downsampled_df[y.name].reset_index(drop=True)
    return X_downsampled, y_downsampled

def z_score_standardization(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return pd.DataFrame(X_standardized, columns=X.columns)

scale = 'Anxiety'
X, y = load_data(scale, '../data/label.csv', '../data/features.csv')
X = X[[' eye_lmk_Z_4_var',' eye_lmk_Z_4_min']]
X, y = random_down_sampling(X, y, 42)
X = z_score_standardization(X).values
y = y.values

cmap = plt.cm.RdYlBu
plot_step = 0.01  # fine step width for decision surface contours
RANDOM_SEED = 13  # fix the seed on each iteration
plot_step_coarser = 0.5
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14

model = RandomForestClassifier(n_estimators=100, 
                               max_depth=50,   # 例如，将深度限制为10
                               min_samples_split=10,   # 节点继续分裂所需的最小样本数
                               min_samples_leaf=5,     # 叶子节点所需的最小样本数
                               max_features='auto',    # 分裂时考虑的最大特征数
                               n_jobs=-1)
# Standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
# Train
model.fit(X, y)

scores = model.score(X, y)
model_title = str(type(model)).split(".")[-1][:-2][: -len("Classifier")]

model_details = model_title
if hasattr(model, "estimators_"):
    model_details += " with {} estimators".format(len(model.estimators_))
print(model_details + " with features has a score of", scores)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
)

if isinstance(model, DecisionTreeClassifier):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap)
else:
    estimator_alpha = 1.0 / len(model.estimators_)
    for tree in model.estimators_:
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

xx_coarser, yy_coarser = np.meshgrid(
    np.arange(x_min, x_max, plot_step_coarser),
    np.arange(y_min, y_max, plot_step_coarser),
)
Z_points_coarser = model.predict(
    np.c_[xx_coarser.ravel(), yy_coarser.ravel()]
).reshape(xx_coarser.shape)
cs_points = plt.scatter(
    xx_coarser,
    yy_coarser,
    s=15,
    c=Z_points_coarser,
    cmap=cmap,
    edgecolors="none",
)

# Plot the training points, these are clustered together and have a
# black outline
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=ListedColormap(["r", "y", "b"]),
    edgecolor="k",
    s=20,
)
plt.axis("tight")
plt.savefig(f'../draw/decisionSurfaces_{scale}.png',dpi=600,bbox_inches='tight', transparent=True)