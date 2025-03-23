import numpy as np
import torch
import logging
from tqdm import tqdm
from game_env import Game2048
from dqn_agent import DQNAgent

import os
from config import TEMP_MODEL_DIR, FINAL_MODEL_DIR, LOG_DIR


def test_model(model_path: str, test_times: int, log_file: str):
    # 先删除原有的日志文件
    if os.path.exists(log_file) and os.path.isfile(log_file):
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
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0.0  # 关闭探索
    for e in tqdm(range(test_times)):
        env = Game2048()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            env.move(["up", "down", "left", "right"][action])
            state = env.get_state()
            done = env.game_over()

        logger.info(
            f"Number: {e+1}/{test_times}\t"
            f"Score: {env.score}\t"
            f"Max Tile: {np.max(env.grid)}\t"
            f"Steps: {env.steps}"
        )


def main():
    # 选择模型
    m = input("Final model or temp model? [F/t]:").strip() or "f"
    model_path = ""
    if m.lower() == "f":
        # 获取目录内容
        all_entries = os.listdir(FINAL_MODEL_DIR)
        # 过滤 .pth 文件并排序
        pth_files = []
        for entry in all_entries:
            full_path = os.path.join(FINAL_MODEL_DIR, entry)
            if os.path.isfile(full_path) and entry.lower().endswith(".pth"):
                pth_files.append(entry)

        # 从大到小，从新到旧
        pth_files.sort(reverse=True)

        # 输出结果
        if not pth_files:
            raise FileNotFoundError(
                f"Directory {os.path.abspath(FINAL_MODEL_DIR)} does not contain any .pth file"
            )
        else:
            print(f"Found the following .pth files in the {FINAL_MODEL_DIR}:\n")
            for idx, filename in enumerate(pth_files, 1):
                print(f"\t{idx}. {filename}")
            print()
            num = int(input("Input number [1]: ") or "1")
            if num > len(pth_files):
                raise ValueError("Invalid number.")
            model_path = os.path.join(FINAL_MODEL_DIR, pth_files[num - 1])
    elif m.lower() == "t":
        ch = int(input("Input checkpoint [100]: ").strip() or "100")
        model_path = os.path.join(TEMP_MODEL_DIR, f"checkpoint_{ch}.pth")
    else:
        raise ValueError("Invalid option. Only f or t.")
    print(f"Model: \033[94m{os.path.abspath(model_path)}\033[0m")
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
        agent.model.load_state_dict(torch.load(model_path))
        agent.epsilon = 0.0  # 关闭探索

        if not env.game_over():
            state = env.get_state()
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            direction = ["up", "down", "left", "right"][action]
            print("\n\033[92m" + direction + "\033[0m\n")
        else:
            print("\n\033[91mGame Over!\033[0m\n")


if __name__ == "__main__":
    main()
