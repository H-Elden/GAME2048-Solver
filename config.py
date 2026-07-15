TEMP_MODEL_DIR = "temp_model"
FINAL_MODEL_DIR = "final_modal"
LOG_DIR = "log"


# 超参数
update_target_freq = 20  # 目标网络更新频率
maxlen = 10000  # 经验回放缓冲区大小
replay_beta = 0.4  # 初始重要性采样参数
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_min = 0.05  # 最小探索率
learning_rate = 0.0001  # 学习率
batch_size = 128  # 批量大小
