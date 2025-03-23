# frequency distribution histogram
# 频数分布直方图

import matplotlib.pyplot as plt
import numpy as np
import math
import os


def save_frequency(scores: list, time: str, chart_dir: str):
    """
    将"Score"数据绘制成柱状图并保存为图片
    参数：
        episodes: 训练编号
        scores: 数据列表
        time: 训练时间，用于副标题
    """
    # 计算新的横轴范围（整百扩展）
    min_score = min(scores)
    max_score = max(scores)
    new_min = math.floor(min_score / 100) * 100
    new_max = math.ceil(max_score / 100) * 100

    # 生成等距分箱（20个区间，21个边界点）
    bins = np.linspace(new_min, new_max, 21, dtype=int)

    # 创建画布和坐标轴（显式使用 fig 和 ax）
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制直方图
    n, bins, patches = ax.hist(
        scores, bins=bins, color="skyblue", edgecolor="black", alpha=0.8
    )

    plt.xticks()
    # 设置横轴刻度和标签（显示所有整数边界点）
    ax.set_xticks(bins, [f"{int(x)}" for x in bins], fontsize=10)
    # 设置坐标轴标签
    ax.set_xlabel("Score Section", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # 设置横轴范围
    ax.set_xlim(new_min, new_max)

    # 添加主标题和副标题（使用 fig.text 固定位置）
    fig.text(
        0.5,
        0.93,
        "Score Frequency Histogram",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.88,
        f"Episodes: {len(scores)}, Time: {time}",
        ha="center",
        fontsize=10,
        color="gray",
    )
    # 添加平均数和最大值的统计结果
    fig.text(
        0.97,
        0.88,
        f"Max: {max(scores)}, Average: {round(sum(scores)/len(scores))}",
        ha="right",
        fontsize=14,
        color="black",
    )

    # 在每个柱子上方显示频率值
    bin_centers = (bins[:-1] + bins[1:]) / 2  # 计算每个组的中心位置
    for i in range(len(n)):
        if n[i] > 0:  # 只显示非零频率
            plt.text(
                bin_centers[i],
                n[i] + 0.5,
                int(n[i]),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 设置网格
    ax.grid(True, axis="y", alpha=0.3)

    # 自动调整坐标轴范围
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)  # 为标题预留空间

    # 保存为图片
    pic_path = os.path.join(chart_dir, f"Score_frequency_histogram_{time}.png")
    fig.savefig(pic_path, dpi=300)

    # 显示图表
    # fig.show()
