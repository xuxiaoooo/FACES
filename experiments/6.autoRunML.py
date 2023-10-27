import subprocess

# 定义所有的选项
models = ["ada", "rf", "et", "xgb", "ada_b", "rf_b", "et_b", "xgb_b"]
scales = ["Depression", "Anxiety", "Stress"]
cluster = ["Cluster_2", "Cluster_3"]
categories = [2, 3]

# 循环执行命令
# for model in models:
#     for scale in scales:
#         for category in categories:
#             # 构建命令行字符串
#             cmd = f"python 5.machineL.py --model {model} --scale {scale} --category {category}"
#             # 使用 subprocess 执行命令
#             subprocess.run(cmd, shell=True)
for model in models:
    for scale in cluster:
        cmd = f"python 5.machineL.py --model {model} --scale {scale} --category {scale.split('_')[1]}"
        subprocess.run(cmd, shell=True)
