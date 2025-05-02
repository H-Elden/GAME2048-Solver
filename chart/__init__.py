import re
import os
from .scatter import save_scatter
from .histogram import save_histogram
from .frequency import save_frequency
from .pie import save_pie_chart

PIC_DIR = "pic"


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


def save_train_chart(log_path: str, time: str):
    scores, max_tiles, steps_list = load_data(log_path)
    # 创建文件夹
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


def save_test_chart(log_path: str, time: str):
    scores, max_tiles, steps_list = load_data(log_path)
    # 创建文件夹
    current_dir = os.path.dirname(__file__)
    chart_dir = os.path.join(current_dir, PIC_DIR, time, "test")
    os.makedirs(chart_dir, exist_ok=True)
    save_histogram(max_tiles, time, chart_dir)
    save_pie_chart(max_tiles, time, chart_dir)
    save_frequency(scores, time, chart_dir)
    print(
        f"Testing charts has been saved to \033[94m{os.path.abspath(chart_dir)}\033[0m folder."
    )


if __name__ == "__main__":
    save_train_chart("output.log", "20250322_2123")
