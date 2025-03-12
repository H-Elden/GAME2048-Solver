# 把 Max Tile 做成柱状图

import re
from collections import Counter
import matplotlib.pyplot as plt

# 初始化数据存储列表
max_tiles = []

# 正则表达式模式匹配
pattern = r"Max Tile: (\d+)"

# 读取日志文件
with open("output.log", "r") as f:
    for line in f:
        # 使用正则表达式提取 Max Tile 数值
        match = re.search(pattern, line)
        if match:
            max_tiles.append(int(match.group(1)))

# 统计每个 Max Tile 值的出现次数
max_tile_counts = Counter(max_tiles)

# 提取键（Max Tile 值）和值（出现次数）
tiles = sorted(max_tile_counts.keys())  # 按从小到大排序
counts = [max_tile_counts[tile] for tile in tiles]  # 按排序后的顺序获取次数

# 创建画布和坐标轴
plt.figure(figsize=(10, 6))

# 绘制柱状图
plt.bar(range(len(tiles)), counts, color="skyblue", edgecolor="black", alpha=0.8)

# 设置横轴刻度和标签
plt.xticks(range(len(tiles)), tiles)  # 横轴刻度为 tiles 的值

# 添加图例和标签
plt.title("Max Tile histogram", fontsize=14)
plt.xlabel("Max Tile", fontsize=12)
plt.ylabel("times", fontsize=12)

# 在柱子上方显示具体数值
for x, y in zip(range(len(tiles)), counts):
    plt.text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=10)

# 设置网格
plt.grid(True, axis="y", alpha=0.3)

# 自动调整坐标轴范围
plt.tight_layout()

# 保存为图片
# plt.savefig("Max_Tile_histogram.png", dpi=300)

# 显示图表
plt.show()
