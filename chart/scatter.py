# 散点图 scatter diagram

import matplotlib.pyplot as plt
import os


def save_scatter(data: list, label: str, time: str, chart_dir: str):
    """
    绘制散点图并保存为图片
    参数：
        data: 数据列表
        label: 数据标签，y轴标签: "Score", "Max Tile", "Steps"
        time: 训练时间，用于副标题
    """
    dic = {
        "Score": {"color": "blue", "marker": "o"},
        "Max Tile": {"color": "red", "marker": "s"},
        "Steps": {"color": "green", "marker": "^"},
    }
    # 创建画布和坐标轴（显式使用 fig 和 ax）
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制散点图
    ax.scatter(
        [x for x in range(1, len(data) + 1)],
        data,
        label=label,
        color=dic[label]["color"],
        marker=dic[label]["marker"],
        alpha=0.7,
    )
    # 设置坐标轴标签
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(label, fontsize=12)

    # 添加主标题和副标题（使用 fig.text 固定位置）
    fig.text(
        0.5,
        0.93,
        f"{label} Scatter Diagram",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.88,
        f"Episodes: {len(data)}, Time: {time}",
        ha="center",
        fontsize=10,
        color="gray",
    )

    # 图例
    # ax.legend(fontsize=10)

    # 设置网格
    ax.grid(True, alpha=0.3)

    # 自动调整坐标轴范围
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)  # 为标题预留空间

    # 保存为图片
    pic_path = os.path.join(chart_dir, f"{label}_scatter_diagram_{time}.png")
    fig.savefig(pic_path, dpi=300)

    # 显示图表
    # fig.show()
