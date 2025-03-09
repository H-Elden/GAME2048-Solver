import numpy as np
import torch
from game_env import Game2048
from dqn_agent import DQNAgent


def train(episodes=1000, batch_size=32):
    # 初始化环境和代理
    agent = DQNAgent(state_size=16, action_size=4)
    update_target_freq = 10  # 目标网络更新频率

    for e in range(episodes):
        env = Game2048()
        state = env.grid.flatten()  # 将棋盘展平为16维向量
        total_reward = 0
        done = False
        step_count = 0  # 步数计数器

        while not done:
            action = agent.act(state)  # 选择动作
            prev_score = env.score
            direction = ["up", "down", "left", "right"][action]
            env.move(direction)  # 执行动作
            next_state = env.grid.flatten()
            reward = env.score - prev_score  # 计算即时奖励
            done = env.game_over()

            # 保存经验
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1  # 每次动作后计数

            if done:
                print(
                    f"Episode: {e+1}/{episodes}, "
                    f"Score: {env.score}, "
                    f"Max Tile: {np.max(env.grid)}, "
                    f"Steps: {env.steps}/{step_count}, "  # 有效步数 / 总步数
                    f"Epsilon: {agent.epsilon:.2f}"
                )

            agent.replay(batch_size)  # 经验回放训练

        # 定期更新目标网络
        if e % update_target_freq == 0:
            agent.update_target_model()

    # 保存训练好的模型
    torch.save(agent.model.state_dict(), "2048_dqn.pth")


if __name__ == "__main__":
    train(episodes=50)  # 开始训练
