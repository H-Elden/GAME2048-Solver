import torch
import tkinter as tk
from game_env import Game2048
from dqn_agent import DQNAgent

import os
from common import select_model


class GameGUI:
    def __init__(self, master, agent, refresh=200):
        self.master = master
        self.agent = agent
        self.refresh = refresh
        self.env = Game2048()

        # 定义不同数字对应的颜色
        self.colors = {
            0: "#CDC1B3",
            2: "#EEE4DA",
            4: "#ECE0C8",
            8: "#F2B179",
            16: "#F59563",
            32: "#F57C5F",
            64: "#F65D3B",
            128: "#EDCE71",
            256: "#EDCC61",
            512: "#ECC850",
            1024: "#EDC53F",
            2048: "#EEC22E",
            4096: "#EEC22E",
            8192: "#EEC22E",
            16384: "#EEC22E",
            32768: "#EEC22E",
            65536: "#EEC22E",
        }

        # 创建图形界面元素
        self.create_widgets()
        self.step_count = 0  # 初始化步数计数器
        self.update_gui()
        self.play_game()  # 开始游戏

    def create_widgets(self):
        # 创建棋盘格
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)
        self.labels = []
        for i in range(4):
            row = []
            for j in range(4):
                label = tk.Label(
                    self.frame,
                    text="",
                    bg=self.colors[0],
                    width=4,
                    height=2,
                    font=("Arial", 24, "bold"),
                )
                label.grid(row=i, column=j, padx=5, pady=5)
                row.append(label)
            self.labels.append(row)

        # 分数显示
        self.score_label = tk.Label(
            self.master, text="Score: 0\tSteps: 0", font=("Arial", 16)
        )
        self.score_label.pack(pady=10)

    def update_gui(self):
        # 更新棋盘显示
        for i in range(4):
            for j in range(4):
                val = self.env.grid[i][j]
                self.labels[i][j].config(
                    text=str(val) if val != 0 else "",
                    bg=self.colors.get(val, "#3C3A32"),
                )
        self.score_label.config(
            text=f"Score: {self.env.score}\tSteps: {self.env.steps}"
        )
        self.master.update()

    def play_game(self):
        while not self.env.game_over():
            state = self.env.get_state()  # 使用新状态表示
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break

            action = self.agent.act(state, valid_actions)
            self.env.move(action)
            self.update_gui()
        print(f"Game Over! Final Score: \033[92m{self.env.score}\033[0m")


if __name__ == "__main__":
    model_path = select_model()
    print(f"Model: \033[94m{os.path.abspath(model_path)}\033[0m")

    # 加载训练好的模型
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.epsilon = 0.0  # 关闭探索

    # 输入gui刷新率
    refresh = int(input("Input GUI refresh rate [200]: ").strip() or "200")

    # 启动图形界面
    root = tk.Tk()
    root.title("2048 DQN Agent (CNN)")
    gui = GameGUI(root, agent, refresh=refresh)
    root.mainloop()
