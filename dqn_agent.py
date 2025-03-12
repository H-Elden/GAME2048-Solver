import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class DQN(nn.Module):
    """CNN网络结构"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道1，输出32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Linear(512, 4)  # 输出4个动作的Q值
        )

    def forward(self, x):
        x = self.conv(x)  # 输入形状: (batch,1,4,4)
        x = x.reshape(x.size(0), -1)  # 展平为 (batch, 64*4*4)
        return self.fc(x)


class DQNAgent:
    def __init__(self):
        self.action_size = 4
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区

        # 超参数设置
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.999  # 探索率衰减率
        self.learning_rate = 0.0001  # 学习率

        self.batch_size = 128

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
        self.memory.append((state, action, reward, next_state, done))

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
        minibatch = random.sample(self.memory, self.batch_size)

        # 转换数据维度 (batch,4,4,1) -> (batch,1,4,4)
        states = (
            torch.FloatTensor(np.array([x[0] for x in minibatch]))
            .permute(0, 3, 1, 2)
            .to(device)
        )
        actions = torch.LongTensor([x[1] for x in minibatch]).to(device)
        rewards = torch.FloatTensor([x[2] for x in minibatch]).to(device)
        next_states = (
            torch.FloatTensor(np.array([x[3] for x in minibatch]))
            .permute(0, 3, 1, 2)
            .to(device)
        )
        dones = torch.FloatTensor([x[4] for x in minibatch]).to(device)

        # 计算当前Q值和目标Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
