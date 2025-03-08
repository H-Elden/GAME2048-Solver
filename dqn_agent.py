import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # 定义三层全连接网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # 输入层到隐藏层
            nn.ReLU(),
            nn.Linear(128, 128),  # 隐藏层到隐藏层
            nn.ReLU(),
            nn.Linear(128, output_dim),  # 隐藏层到输出层
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区

        # 超参数设置
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减率
        self.learning_rate = 0.001  # 学习率

        # 初始化模型和优化器
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()  # 初始化目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        # 将目标网络参数更新为当前网络参数
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # 保存经验到记忆库
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon-greedy策略选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机探索
        state = torch.FloatTensor(state).to(device)
        act_values = self.model(state)
        return torch.argmax(act_values).item()  # 选择最优动作

    def replay(self, batch_size):
        # 经验回放训练过程
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        # 转换数据格式
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(device)

        # 计算当前Q值和目标Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
