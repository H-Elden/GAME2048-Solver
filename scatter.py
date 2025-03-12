# 散点图 scatter diagram

import re
import matplotlib.pyplot as plt

# 初始化数据存储列表
episodes = []
scores = []
max_tiles = []
steps_list = []

# 正则表达式模式匹配
pattern = r"Score: (\d+)\tMax Tile: (\d+)\tSteps: (\d+)"

# 读取日志文件
with open("output.log", "r") as f:
    for line in f:
        # 使用正则表达式提取数值
        match = re.search(pattern, line)
        if match:
            scores.append(int(match.group(1)))
            max_tiles.append(int(match.group(2)))
            steps_list.append(int(match.group(3)))

# 生成横轴数据（行数/Episode序号）
episodes = list(range(1, len(scores) + 1))

# 创建画布和坐标轴
plt.figure(figsize=(12, 6))

# 绘制三条折线
# plt.plot(episodes, scores, label="Score", color="blue", linestyle="-")
# plt.plot(episodes, max_tiles, label="Max Tile", color="red", linestyle="--")
# plt.plot(episodes, steps_list, label="Steps", color="green", linestyle="-.")

# 绘制散点图
plt.scatter(episodes, scores, label="Score", color="blue", marker="o", alpha=0.7)
# plt.scatter(episodes, max_tiles, label="Max Tile", color="red", marker="s", alpha=0.7)
# plt.scatter(episodes, steps_list, label="Steps", color="green", marker="^", alpha=0.7)

# 添加图例和标签
plt.title("5000 Episodes Scatter Diagram", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Max Tile", fontsize=12)
plt.legend(fontsize=10)

# 设置网格
plt.grid(True, alpha=0.3)

# 自动调整坐标轴范围
plt.tight_layout()

# 保存为图片
# plt.savefig("MaxTile_scatter_diagram.png", dpi=300)

# 显示图表
plt.show()
