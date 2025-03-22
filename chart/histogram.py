# 把 Max Tile 做成柱状图

from collections import Counter
import matplotlib.pyplot as plt
import os


def save_histogram(episodes: list[int], max_tiles: list, time: str, chart_dir: str):
    """
    将"Max Tile"数据绘制成柱状图并保存为图片
    参数：
        episodes: 训练编号
        max_tiles: 数据列表
        time: 训练时间，用于副标题
    """
    # 统计每个 Max Tile 值的出现次数
    max_tile_counts = Counter(max_tiles)

    # 提取键（Max Tile 值）和值（出现次数）
    tiles = sorted(max_tile_counts.keys())  # 按从小到大排序
    counts = [max_tile_counts[tile] for tile in tiles]  # 按排序后的顺序获取次数

    # 创建画布和坐标轴（显式使用 fig 和 ax）
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    ax.bar(range(len(tiles)), counts, color="skyblue", edgecolor="black", alpha=0.8)

    # 设置横轴刻度和标签
    ax.set_xticks(range(len(tiles)), tiles)
    # 设置坐标轴标签
    ax.set_xlabel("Max Tile", fontsize=12)
    ax.set_ylabel("Times", fontsize=12)

    # 添加主标题和副标题（使用 fig.text 固定位置）
    fig.text(
        0.5,
        0.93,
        "Max Tile histogram",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.88,
        f"Episodes: {len(episodes)}, Time: {time}",
        ha="center",
        fontsize=10,
        color="gray",
    )

    # 在柱子上方显示具体数值
    for x, y in zip(range(len(tiles)), counts):
        ax.text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=10)

    # 设置网格
    ax.grid(True, axis="y", alpha=0.3)

    # 自动调整坐标轴范围
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)  # 为标题预留空间

    # 保存为图片
    pic_path = os.path.join(chart_dir, f"Max_Tile_histogram_{time}.png")
    fig.savefig(pic_path, dpi=300)

    # 显示图表
    # fig.show()
