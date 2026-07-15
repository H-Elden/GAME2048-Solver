# 2048 DQN Solver

基于 **深度 Q 网络（Deep Q-Network）** 的 2048 游戏 AI 求解器，使用 PyTorch 实现。

## 特性

- **CNN 网络**：保留 4x4 棋盘的空间结构，比全连接网络更擅长捕捉相邻方块关系
- **双重 DQN（Double DQN）**：当前网络 + 目标网络分离，稳定 Q 值估计
- **优先级经验回放（Prioritized Experience Replay）**：按 TD 误差优先采样高价值经验，提高训练效率
- **有效动作屏蔽**：只在当前合法动作中选择，避免无意义的无效尝试
- **多维度复合奖励**：分数变化 + 最大方块指数奖励 + 空位奖励 + 单调性奖励，全面引导策略学习
- **完整训练管线**：训练 → 日志记录 → 统计图表生成 → 模型测试，一站式自动化
- **GUI 可视化**：Tkinter 图形界面实时展示 AI 游戏过程

## 环境要求

- Python 3.13+
- PyTorch
- NumPy
- Matplotlib
- tqdm

```bash
pip install torch numpy matplotlib tqdm
```

## 快速开始

### 训练模型

```bash
python train.py
```

交互式选项：
- 输入训练轮数（episode）
- 是否生成统计图表
- 是否自动测试并生成测试报告

训练过程中每 100 轮自动保存检查点到 `temp_model/`，训练结束保存最终模型到 `final_modal/`。

### 查看 AI 玩游戏

```bash
python gui.py
```

选择模型文件后，GUI 窗口将展示 AI 实时决策过程。

### 手动测试

```bash
python test.py
```

支持两种模式：
- **批量测试**：加载模型运行 N 次，统计得分分布
- **单步推理**：手动输入 4x4 棋盘，查看模型推荐动作

## 项目结构

```
DQN-game2048/
├── config.py              # 超参数与路径配置
├── game_env.py            # 2048 游戏环境
├── dqn_agent.py           # DQN 网络 + Agent + 优先级经验回放
├── train.py               # 训练主程序 + 奖励函数
├── test.py                # 模型测试与单局面推理
├── gui.py                 # 图形界面
├── chart/                 # 图表生成模块
│   ├── __init__.py        # 日志解析、图表分发
│   ├── scatter.py         # 散点图（Score/Max Tile/Steps）
│   ├── histogram.py       # Max Tile 柱状图
│   ├── frequency.py       # Score 频数分布直方图
│   └── pie.py             # Max Tile 饼图
├── final_modal/           # 训练好的最终模型
├── temp_model/            # 训练检查点
├── log/                   # 训练/测试日志
└── chart/pic/             # 生成的统计图表
```

## 核心算法

### 网络结构

```
输入 (1, 4, 4)
  → Conv2d(1→32, 3×3) + ReLU
  → Conv2d(32→64, 3×3) + ReLU
  → Conv2d(64→64, 3×3) + ReLU
  → Linear(1024→512) + ReLU
  → Linear(512→4)
```

### 奖励函数

奖励 = **分数变化** + **最大方块指数奖励** + **空位平方差奖励** + **单调性奖励 × 5**

- 分数变化：鼓励有效合并
- 最大方块奖励：强化合成大数字，采用 `max_tile × (1 + 0.1 × log₂(max_tile))` 指数增长
- 空位奖励：鼓励保持棋盘空间
- 单调性奖励：引导数值有序排列

### 探索策略

Epsilon-greedy 策略，探索率 1.0 → 0.05 线性衰减（前 90% 训练周期），仅从有效动作中随机探索。

## 训练效果

在 1000 次训练的模型上测试 100 局：

| 指标 | 数值 |
|------|------|
| 最高分 | 10632 |
| 平均分 | 3293 |
| 合成 1024 | 2% |
| 合成 512 | 18% |
| 合成 256 | 49% |

> 详细训练过程和技术分析见 [ARCHITECTURE.md](./ARCHITECTURE.md)

## 配置参数

所有超参数集中在 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `learning_rate` | 0.0001 | 学习率 |
| `batch_size` | 128 | 批量大小 |
| `maxlen` | 10000 | 经验回放缓冲区容量 |
| `epsilon_min` | 0.05 | 最小探索率 |
| `update_target_freq` | 20 | 目标网络更新频率 |

## License

MIT License
