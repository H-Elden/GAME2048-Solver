import re
import os
from .scatter import save_scatter
from .histogram import save_histogram
from .frequency import save_frequency

PIC_DIR = "pic"


def load_data(file_path: str):
    # 初始化数据存储列表
    episodes = []
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

    # 生成横轴数据（行数/Episode序号）
    episodes = list(range(1, len(scores) + 1))
    return episodes, scores, max_tiles, steps_list


def save_chart(log_path: str, time: str):
    episodes, scores, max_tiles, steps_list = load_data(log_path)
    # 创建文件夹
    current_dir = os.path.dirname(__file__)
    chart_dir = os.path.join(current_dir, PIC_DIR, time)
    os.makedirs(chart_dir, exist_ok=True)
    save_scatter(episodes, scores, "Score", time, chart_dir)
    save_scatter(episodes, max_tiles, "Max Tile", time, chart_dir)
    save_scatter(episodes, steps_list, "Steps", time, chart_dir)
    save_histogram(episodes, max_tiles, time, chart_dir)
    save_frequency(episodes, scores, time, chart_dir)
    print(
        f"Charts has been saved to \033[94m{os.path.abspath(chart_dir)}\033[0m folder."
    )


if __name__ == "__main__":
    save_chart("output.log", "20250322_2123")
