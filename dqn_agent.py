import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

from config import (
    maxlen,
    gamma,
    epsilon,
    epsilon_min,
    epsilon_decay,
    learning_rate,
    replay_beta,
    batch_size,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class DQN(nn.Module):
    """CNN网络结构"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 输入1通道，输出32通道，3*3卷积核，填充1层0保证图大小不变
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输入32通道，输出64通道
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 输入64通道，输出64通道
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Linear(512, 4)  # 输出4个动作的Q值
        )

    # 前向传播流程
    def forward(self, x):
        x = self.conv(x)  # 输入形状: (batch,1,4,4)
        x = x.reshape(x.size(0), -1)  # 展平为 (batch, 64*4*4)
        return self.fc(x)


class DQNAgent:
    def __init__(self):
        self.action_size = 4
        # 优先回放缓冲区
        self.memory = PrioritizedReplayBuffer(capacity=maxlen, alpha=0.6)

        # 超参数设置
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 初始探索率
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减率
        self.learning_rate = learning_rate  # 学习率
        self.beta = replay_beta  # 初始重要性采样参数

        self.batch_size = batch_size  # 批量大小

        # 初始化模型和优化器
        self.model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.update_target_model()  # 初始化目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        """将目标网络参数更新为当前网络参数"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """保存经验到记忆库"""
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        """智能动作选择：只在有效动作中探索/利用"""
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions) if valid_actions else 0
        else:
            state = (
                torch.FloatTensor(state)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device)
            )  # 调整维度
            with torch.no_grad():
                q_values = self.model(state)
            q_np = q_values.cpu().numpy().flatten()
            return valid_actions[
                np.argmax(q_np[valid_actions])
            ]  # 只在有效动作中选择最大值

    def replay(self):
        """经验回放训练过程"""
        if len(self.memory) < self.batch_size:
            return

        # 采样并获取权重
        samples, indices, weights = self.memory.sample(self.batch_size, beta=self.beta)
        states, actions, rewards, next_states, dones = zip(*samples)

        # 转换数据格式
        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = (
            torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(device)
        )
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # 计算当前Q值和目标Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算TD误差并更新优先级
        td_errors = torch.abs(current_q.squeeze() - target_q).cpu().detach().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-5)  # 防止零误差

        # 计算加权损失
        loss = (weights * (current_q.squeeze() - target_q) ** 2).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 逐步增加beta（从0.4到1.0）
        self.beta = min(1.0, self.beta + 0.001)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # 存储每个样本的优先级

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        """添加经验时初始化最大优先级"""
        max_prio = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        """根据优先级采样"""
        # 计算概率分布
        priorities = np.array(self.priorities)
        probs = priorities**self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, new_priorities):
        """更新采样样本的优先级（基于新的TD误差）"""
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio
