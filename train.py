import numpy as np
import torch
from game_env import Game2048
from dqn_agent import DQNAgent
import logging
from tqdm import tqdm
from chart import save_train_chart, save_test_chart
from datetime import datetime
from test import test_model
from config import TEMP_MODEL_DIR, FINAL_MODEL_DIR, LOG_DIR

import os


def calculate_reward(env, prev_score, prev_max_tile, prev_empty_cells):
    # 1. 基础得分奖励（合并方块的即时收益）
    score_reward = env.score - prev_score

    # 2. 最大方块指数奖励（强化大数字的合成）
    current_max = np.max(env.grid)
    max_tile_reward = 0
    if current_max > prev_max_tile:
        # 128 -> 128 * 1.7; 256 -> 256 * 1.8
        max_tile_reward = current_max * (1 + 0.1 * np.log2(current_max))

    # 3. 空位数量奖励（鼓励保留更多空位）
    empty_cells = np.sum(env.grid == 0)
    empty_reward = (empty_cells**2) - (prev_empty_cells**2)  # 奖励空位数量平方的增长

    # 4. 方块有序性奖励（鼓励数值按行/列有序排列）
    monotonicity = calculate_monotonicity(env.grid)
    mono_reward = monotonicity * 5  # 乘以系数平衡奖励量级

    # # 5. 无效移动惩罚（施加强惩罚阻止无效动作）
    # if env.grid_equal_after_move:  # 需在env中添加此判断
    #     invalid_penalty = -10.0
    # else:
    #     invalid_penalty = 0.0

    # 6. 总奖励组合
    total_reward = score_reward + max_tile_reward + empty_reward + mono_reward
    return total_reward


def calculate_monotonicity(grid):
    score = 0
    for i in range(4):
        # 行单调性
        row = grid[i, :]
        row_score = 0
        for j in range(3):
            if row[j] >= row[j + 1]:
                row_score += 1
        score += row_score
        # 列单调性（需转置）
        col = grid[:, i]
        col_score = 0
        for j in range(3):
            if col[j] >= col[j + 1]:
                col_score += 1
        score += col_score
    # score 不超过24
    return score


def train(model_path: str, episodes=5000, log_file=None):
    """
    训练DQN模型

    """
    if log_file:
        logger = logging.getLogger("train")
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
            f"The training log will be saved to \033[94m{os.path.abspath(log_file)}\033[0m"
        )
    # 初始化环境和代理
    agent = DQNAgent()
    update_target_freq = 50  # 目标网络更新频率

    for e in tqdm(range(episodes)):
        env = Game2048()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            prev_score = env.score
            prev_max_tile = np.max(env.grid)
            prev_empty_cells = np.sum(env.grid == 0)
            env.move(["up", "down", "left", "right"][action])
            next_state = env.get_state()
            reward = calculate_reward(env, prev_score, prev_max_tile, prev_empty_cells)
            done = env.game_over()

            # 保存经验
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # 经验回放训练
            agent.replay()

        if log_file:
            logger.info(
                f"Episode: {e+1}/{episodes}\t"
                f"Score: {env.score}\t"
                f"Max Tile: {np.max(env.grid)}\t"
                f"Steps: {env.steps}\t"
                f"Epsilon: {agent.epsilon:.2f}"
            )
        # 衰减探索率
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = 1.0 - (1.0 - agent.epsilon_min) * (e / (episodes * 0.9))

        # 定期更新目标网络
        if e % update_target_freq == 0:
            agent.update_target_model()

        # 每100轮保存一次，保存模型到 temp_model 文件夹
        if (e + 1) % 100 == 0:
            checkpoint_path = os.path.join(TEMP_MODEL_DIR, f"checkpoint_{e+1}.pth")
            torch.save(agent.model.state_dict(), checkpoint_path)

    torch.save(agent.model.state_dict(), model_path)
    print(
        f"The final model has been saved to \033[94m{os.path.abspath(model_path)}\033[0m"
    )


if __name__ == "__main__":
    episodes = int(input("Input episodes: "))
    chart = input("Generate statistical chart? [Y/n]: ").strip() or "y"
    test = input("Automatically test the final model? [Y/n]: ").strip() or "y"
    now_time = datetime.now().strftime("%Y%m%d_%H%M")

    # 确保日志文件夹存在
    subdir = os.path.join(LOG_DIR, f"{now_time}")
    os.makedirs(subdir, exist_ok=True)
    train_log_file = os.path.join(subdir, f"train.log")
    test_log_file = os.path.join(subdir, f"test.log")

    # 模型地址
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(FINAL_MODEL_DIR, f"2048_dqn_final_{now_time}.pth")

    # 开始训练
    train(model_path, episodes=episodes, log_file=train_log_file)

    # 绘制并保存训练统计图
    if chart.lower() == "y":
        save_train_chart(train_log_file, now_time)

    if test.lower() == "y":
        test_model(model_path, 100, test_log_file)
        save_test_chart(test_log_file, now_time)
