import numpy as np
import torch
from game_env import Game2048
from dqn_agent import DQNAgent
import logging
from tqdm import tqdm

import os

# 确保 temp_model 文件夹存在
if not os.path.exists("temp_model"):
    os.makedirs("temp_model")

# 配置 logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 设置日志格式
    filename="output.log",  # 指定日志文件
    filemode="w",  # 文件模式：'w' 表示覆盖写入，'a' 表示追加写入
)


def calculate_reward(env, prev_score, prev_max_tile):
    """
    改进的奖励函数：
    - 得分变化奖励
    - 最大数字奖励
    - 步数惩罚
    """
    # 1. 得分变化奖励
    score_reward = env.score - prev_score

    # 2. 最大数字奖励（鼓励合成更大的数字）
    current_max_tile = np.max(env.grid)
    max_tile_reward = 0
    if current_max_tile > prev_max_tile:
        max_tile_reward = current_max_tile * 0.1  # 每合成一个更大的数字，给予额外奖励

    # 3. 步数惩罚（鼓励用更少的步数）
    step_penalty = -1  # 每多走一步，施加一个小的惩罚

    # 4. 空位奖励（鼓励保留更多空位）
    empty_cells = np.sum(env.grid == 0)
    empty_reward = empty_cells * 1

    # 总奖励
    total_reward = score_reward + max_tile_reward + step_penalty + empty_reward
    return total_reward


def train(episodes=5000):
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
            env.move(["up", "down", "left", "right"][action])
            next_state = env.get_state()
            reward = calculate_reward(env, prev_score, prev_max_tile)
            done = env.game_over()

            # 保存经验
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # 经验回放训练
            agent.replay()

        logging.info(
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
            model_path = os.path.join("temp_model", f"checkpoint_{e+1}.pth")
            torch.save(agent.model.state_dict(), model_path)

    torch.save(agent.model.state_dict(), "2048_dqn_cnn_final.pth")


if __name__ == "__main__":
    train()  # 开始训练
