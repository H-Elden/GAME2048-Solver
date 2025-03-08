import torch
import tkinter as tk
from tkinter import font
import numpy as np
from game_env import Game2048
from dqn_agent import DQNAgent


class GameGUI:
    def __init__(self, master, agent):
        self.master = master
        self.agent = agent
        self.env = Game2048()

        # 定义不同数字对应的颜色
        self.colors = {
            0: "#CCC0B3",
            2: "#EEE4DA",
            4: "#EDE0C8",
            8: "#F2B179",
            16: "#F59563",
            32: "#F67C5F",
            64: "#F65E3B",
            128: "#EDCF72",
            256: "#EDCC61",
            512: "#EDC850",
            1024: "#EDC53F",
            2048: "#EDC22E",
        }

        # 创建图形界面元素
        self.create_widgets()
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
        self.score_label = tk.Label(self.master, text="Score: 0", font=("Arial", 16))
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
        self.score_label.config(text=f"Score: {self.env.score}")
        self.master.update()

    def play_game(self):
        # 使用训练好的模型进行游戏
        while not self.env.game_over():
            state = self.env.grid.flatten()
            action = self.agent.act(state)
            direction = ["up", "down", "left", "right"][action]
            self.env.move(direction)
            self.update_gui()
            self.master.after(200)  # 控制移动速度
        print("Game Over! Final Score:", self.env.score)


if __name__ == "__main__":
    # 加载训练好的模型
    agent = DQNAgent(state_size=16, action_size=4)
    agent.model.load_state_dict(torch.load("2048_dqn.pth"))
    agent.epsilon = 0.0  # 关闭探索

    # 启动图形界面
    root = tk.Tk()
    root.title("2048 DQN Agent")
    gui = GameGUI(root, agent)
    root.mainloop()
