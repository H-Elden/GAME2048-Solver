import os
import re
import numpy as np
import torch
import logging
from datetime import datetime
from tqdm import tqdm
from game_env import Game2048
from dqn_agent import DQNAgent
from chart import save_test_chart
from config import LOG_DIR

from common import select_model


def test_model(model_path: str, test_times: int, log_file: str):
    # 先删除原有的日志文件
    if os.path.isfile(log_file):
        os.remove(log_file)
        print(f"Delete file \033[94m{log_file}\033[0m successfully!")
    # 配置日志
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    # 配置 FileHandler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    # 配置 Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    # 添加 Handler 并关闭日志传递
    logger.addHandler(handler)
    logger.propagate = False  # 关闭日志传递到父 Logger
    print(
        f"The testing log will be saved to \033[94m{os.path.abspath(log_file)}\033[0m"
    )

    # 初始化环境和代理
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.epsilon = 0.0  # 关闭探索
    for e in tqdm(range(test_times)):
        env = Game2048()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            env.move(action)
            state = env.get_state()
            done = env.game_over()

        logger.info(
            f"Number: {e+1}/{test_times}\t"
            f"Score: {env.score}\t"
            f"Max Tile: {np.max(env.grid)}\t"
            f"Steps: {env.steps}"
        )


def main():
    model_path = select_model()
    print(f"Model: \033[94m{os.path.abspath(model_path)}\033[0m")

    # 模式选择
    print("\nSelect mode:")
    print("  1. Batch test (multiple episodes with chart & statistics)")
    print("  2. Manual board (single-step inference)")
    mode = input("Choose [1]: ").strip() or "1"

    if mode == "1":
        # 批量测试模式
        test_times = int(input("Input test times [100]: ").strip() or "100")
        now_time = datetime.now().strftime("%Y%m%d_%H%M")

        # 从模型文件名提取训练时间戳，定位 log 目录
        basename = os.path.basename(model_path)
        m = re.search(r"(\d{8}_\d{4})", basename)
        model_time = m.group(1) if m else None
        log_dir = os.path.join(LOG_DIR, model_time) if model_time else os.path.join(LOG_DIR, f"retest_{now_time}")
        os.makedirs(log_dir, exist_ok=True)

        test_log_file = os.path.join(log_dir, f"test_{now_time}.log")
        test_model(model_path, test_times, test_log_file)

        # 图表路径: chart/pic/{model_time}/test/{now_time}/
        if model_time:
            chart_base = os.path.join(os.path.dirname(__file__), "chart")
            chart_test_dir = os.path.join(chart_base, "pic", model_time, "test", now_time)
        else:
            chart_test_dir = None
        save_test_chart(test_log_file, now_time, model_path, chart_dir=chart_test_dir)
        return

    # 模式 2：手动输入盘面，单步推理
    while True:
        # 提示用户输入 4x4 矩阵的元素
        print("Input game board:")
        matrix = []
        for i in range(4):
            while True:
                try:
                    # 获取用户输入的一行数据
                    user_input = input(f"row {i+1}: ").strip()
                    if user_input == "exit":
                        return
                    if not user_input:  # 如果用户直接按回车（空输入）
                        row = [0] * 4
                    else:
                        # 将输入分割并转换为浮点数列表
                        row = list(map(float, user_input.split()))
                        if len(row) > 4:  # 检查是否超出 4 个数字
                            raise ValueError("每行最多只能输入 4 个数字，请重新输入！")
                        # 如果不足 4 个数字，用 0 补齐
                        row.extend([0] * (4 - len(row)))
                    matrix.append(row)
                    break
                except ValueError as e:
                    print(e)

        # 将二维列表转换为 NumPy 数组
        np_matrix = np.array(matrix)

        # 打印生成的 NumPy 矩阵
        print("\n生成的 4x4 NumPy 矩阵为：")
        print(np_matrix)
        # 初始化环境
        env = Game2048(np_matrix)

        # 加载训练好的模型
        agent = DQNAgent()
        agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.epsilon = 0.0  # 关闭探索

        if not env.game_over():
            state = env.get_state()
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            direction = ["up", "down", "left", "right"][action]
            print(f"\n\033[92m{direction}\033[0m\n")
        else:
            print("\n\033[91mGame Over!\033[0m\n")


if __name__ == "__main__":
    main()
