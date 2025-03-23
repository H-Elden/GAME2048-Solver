from collections import Counter
import matplotlib.pyplot as plt
import os


def save_pie_chart(max_tiles: list, time: str, chart_dir: str):
    """
    将"Max Tile"数据绘制成饼状图并保存为图片
    参数：
        episodes: 训练编号
        max_tiles: 数据列表
        time: 训练时间，用于副标题
    """
    # 统计每个 Max Tile 值的出现次数
    max_tile_counts = Counter(max_tiles)

    # 按数值大小降序排列
    sorted_items = sorted(max_tile_counts.items(), key=lambda x: -x[0])
    tiles = [str(item[0]) for item in sorted_items]  # 转换为字符串标签
    counts = [item[1] for item in sorted_items]

    # 创建画布和坐标轴（调整宽高比为正方形）
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制饼状图（自动计算百分比）
    wedges, labels, autotexts = ax.pie(
        counts,
        labels=tiles,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 0.5},
    )

    # 设置主标题和副标题
    fig.text(
        0.5,
        0.88,
        "Max Tile Pie Chart",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.85,
        f"Test times: {len(max_tiles)}, Time: {time}",
        ha="center",
        fontsize=10,
        color="gray",
    )

    # 调整百分比标签显示效果
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)

    # 确保饼图呈正圆形
    ax.axis("equal")
    fig.subplots_adjust(top=0.80)  # 为标题预留空间

    # 保存为图片
    pic_path = os.path.join(chart_dir, f"Max_Tile_pie_chart_{time}.png")
    fig.savefig(pic_path, dpi=300, bbox_inches="tight")

    # 显示图表
    # fig.show()
