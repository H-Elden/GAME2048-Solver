import re
import os
import numpy as np
from collections import Counter
from datetime import datetime
from .scatter import save_scatter
from .histogram import save_histogram
from .frequency import save_frequency
from .pie import save_pie_chart

PIC_DIR = "pic"


def _write_statistics_log(scores, max_tiles, steps_list, log_dir, label, model_path=None):
    """将统计摘要写入 log_dir/statistics.log。首次写入覆盖，后续追加。"""
    n = len(scores)
    if n == 0:
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score_arr = np.array(scores)
    steps_arr = np.array(steps_list)
    tile_counter = Counter(max_tiles)

    log_path = os.path.join(log_dir, "statistics.log")
    is_new = not os.path.exists(log_path)
    mode = 'w' if is_new else 'a'

    lines = []

    # 文件首次创建时写入模型头部信息（移动文件后仍可追溯到来源）
    if is_new and model_path:
        sep = "=" * 80
        lines.append(sep)
        lines.append(f"Model: {os.path.abspath(model_path)}")
        lines.append(f"Training Episodes: {n}")
        lines.append(sep)
        lines.append("")

    sep = "=" * 80
    lines.append(sep)
    lines.append(f"{label} Statistics — Generated at {now_str}")
    lines.append(sep)

    ep_label = "Test Episodes" if label == "Testing" else "Episodes"
    lines.append(f"{ep_label}: {n}")
    lines.append("")

    # Score
    lines.append("Score:")
    lines.append(
        f"  Min: {score_arr.min():,}    Max: {score_arr.max():,}    "
        f"Mean: {score_arr.mean():,.0f}    Median: {np.median(score_arr):,.0f}    "
        f"Std: {score_arr.std():,.0f}"
    )
    lines.append("")

    # Max Tile distribution
    lines.append("Max Tile Distribution:")
    for tile in sorted(tile_counter.keys()):
        count = tile_counter[tile]
        pct = count / n * 100
        lines.append(f"  {tile:>5}: {count:>5}/{n} ({pct:>5.1f}%)")
    lines.append("")

    # Steps
    lines.append("Steps:")
    lines.append(
        f"  Min: {steps_arr.min():,}    Max: {steps_arr.max():,}    "
        f"Mean: {steps_arr.mean():,.0f}"
    )
    lines.append("")

    # Best & Worst
    best_idx = int(np.argmax(score_arr))
    worst_idx = int(np.argmin(score_arr))
    lines.append(
        f"Best  — Episode #{best_idx + 1}  "
        f"Score: {scores[best_idx]:,}  "
        f"Max Tile: {max_tiles[best_idx]}  "
        f"Steps: {steps_list[best_idx]}"
    )
    lines.append(
        f"Worst — Episode #{worst_idx + 1}  "
        f"Score: {scores[worst_idx]:,}  "
        f"Max Tile: {max_tiles[worst_idx]}  "
        f"Steps: {steps_list[worst_idx]}"
    )
    lines.append("")

    with open(log_path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"Statistics log saved to \033[94m{os.path.abspath(log_path)}\033[0m"
    )


def load_data(file_path: str):
    # 初始化数据存储列表
    scores = []
    max_tiles = []
    steps_list = []

    # 正则表达式模式匹配
    pattern = r"Score: (\d+)\tMax Tile: (\d+)\tSteps: (\d+)"

    # 读取日志文件
    with open(file_path, "r") as f:
        for line in f:
            # 使用正则表达式提取数值
            match = re.search(pattern, line)
            if match:
                scores.append(int(match.group(1)))
                max_tiles.append(int(match.group(2)))
                steps_list.append(int(match.group(3)))

    return scores, max_tiles, steps_list


def save_train_chart(log_path: str, time: str, model_path=None):
    scores, max_tiles, steps_list = load_data(log_path)
    # 创建图表文件夹
    current_dir = os.path.dirname(__file__)
    chart_dir = os.path.join(current_dir, PIC_DIR, time, "train")
    os.makedirs(chart_dir, exist_ok=True)
    save_scatter(scores, "Score", time, chart_dir)
    save_scatter(max_tiles, "Max Tile", time, chart_dir)
    save_scatter(steps_list, "Steps", time, chart_dir)
    save_histogram(max_tiles, time, chart_dir)
    save_frequency(scores, time, chart_dir)
    print(
        f"Training charts has been saved to \033[94m{os.path.abspath(chart_dir)}\033[0m folder."
    )
    # 同步生成统计日志
    log_dir = os.path.dirname(os.path.abspath(log_path))
    _write_statistics_log(scores, max_tiles, steps_list, log_dir, "Training", model_path)


def save_test_chart(log_path: str, time: str, model_path=None):
    scores, max_tiles, steps_list = load_data(log_path)
    # 创建图表文件夹
    current_dir = os.path.dirname(__file__)
    chart_dir = os.path.join(current_dir, PIC_DIR, time, "test")
    os.makedirs(chart_dir, exist_ok=True)
    save_histogram(max_tiles, time, chart_dir)
    save_pie_chart(max_tiles, time, chart_dir)
    save_frequency(scores, time, chart_dir)
    print(
        f"Testing charts has been saved to \033[94m{os.path.abspath(chart_dir)}\033[0m folder."
    )
    # 同步生成统计日志
    log_dir = os.path.dirname(os.path.abspath(log_path))
    _write_statistics_log(scores, max_tiles, steps_list, log_dir, "Testing", model_path)


if __name__ == "__main__":
    save_train_chart("output.log", "20250322_2123")
