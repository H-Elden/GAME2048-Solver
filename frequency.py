# frequency distribution histogram
# 频数分布直方图

import re
import matplotlib.pyplot as plt
import numpy as np
import math

# 初始化数据存储列表
scores = []

# 正则表达式模式匹配
pattern = r"Score: (\d+)"

# 读取日志文件
with open("output.log", "r") as f:
    for line in f:
        # 使用正则表达式提取 Score 数值
        match = re.search(pattern, line)
        if match:
            scores.append(int(match.group(1)))

# 计算新的横轴范围（整百扩展）
min_score = min(scores)
max_score = max(scores)
new_min = math.floor(min_score / 100) * 100
new_max = math.ceil(max_score / 100) * 100

# 生成等距分箱（20个区间，21个边界点）
bins = np.linspace(new_min, new_max, 21, dtype=int)

# 创建画布和坐标轴
plt.figure(figsize=(12, 6))

# 绘制直方图
n, bins, patches = plt.hist(
    scores, bins=bins, color="skyblue", edgecolor="black", alpha=0.8
)

# 设置横轴刻度和标签（显示所有整数边界点）
plt.xticks(bins, [f"{int(x)}" for x in bins], fontsize=10)

# 设置横轴范围
plt.xlim(new_min, new_max)

# 添加图例和标签
plt.title("Score Frequency Histogram", fontsize=14)
plt.xlabel("Score Section", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# 在每个柱子上方显示频率值
bin_centers = (bins[:-1] + bins[1:]) / 2  # 计算每个组的中心位置
for i in range(len(n)):
    if n[i] > 0:  # 只显示非零频率
        plt.text(
            bin_centers[i], n[i] + 0.5, int(n[i]), ha="center", va="bottom", fontsize=9
        )

# 设置网格
plt.grid(True, axis="y", alpha=0.3)

# 自动调整布局
plt.tight_layout()

# 保存为图片
# plt.savefig("score_frequency_histogram.png", dpi=300)

# 显示图表
plt.show()
