import numpy as np
import random


class Game2048:
    def __init__(self, initial_grid=None):
        if initial_grid is not None:
            # 检查输入矩阵是否为 4x4
            if isinstance(initial_grid, np.ndarray) and initial_grid.shape == (4, 4):
                self.grid = initial_grid.copy()  # 使用提供的矩阵初始化棋盘
            else:
                raise ValueError("初始棋盘必须是一个 4x4 的 NumPy 数组！")
        else:
            # 初始化4x4棋盘，全零矩阵
            self.grid = np.zeros((4, 4), dtype=int)
            # 游戏开始时随机生成两个数（2或4）
            self.add_random_tile()
            self.add_random_tile()
        self.score = 0  # 得分
        self.steps = 0  # 有效移动步数

    def add_random_tile(self):
        """在空白位置随机添加新数字（90%概率2，10%概率4）"""
        empty = list(zip(*np.where(self.grid == 0)))
        if empty:
            x, y = random.choice(empty)
            self.grid[x][y] = 2 if random.random() < 0.9 else 4

    def get_state(self):
        """优化状态表示：对数处理并保留空间结构"""
        log_grid = np.log2(self.grid + 1e-5)  # 防止log(0)
        return log_grid.reshape(4, 4, 1)  # 输出形状(4,4,1)

    def get_valid_actions(self):
        """返回有效动作索引列表[0-3]（只检查棋盘，不创建新实例）"""
        valid = []
        for idx in range(4):
            if not np.array_equal(self._simulate_move(idx), self.grid):
                valid.append(idx)
        return valid

    def _simulate_move(self, action_idx):
        """模拟移动并返回移动后的棋盘（不修改当前实例、不生成新方块）"""
        direction = ["up", "down", "left", "right"][action_idx]
        if direction == "up":
            return self._move_up(self.grid)
        elif direction == "down":
            return self._move_down(self.grid)
        elif direction == "left":
            return self._move_left(self.grid)
        elif direction == "right":
            return self._move_right(self.grid)

    def game_over(self) -> bool:
        """判断是否还有可行的移动方向"""
        return not self.get_valid_actions()

    def move(self, direction):
        # 执行移动操作，记录原始状态用于比较
        orig_grid = self.grid.copy()
        if direction == "up":
            self.grid = self._move_up(self.grid)
        elif direction == "down":
            self.grid = self._move_down(self.grid)
        elif direction == "left":
            self.grid = self._move_left(self.grid)
        elif direction == "right":
            self.grid = self._move_right(self.grid)

        # 如果棋盘状态改变，则添加新数字
        if not np.array_equal(orig_grid, self.grid):
            self.add_random_tile()
            self.steps += 1  # 有效移动步数增加

    def _move(self, row):
        # 核心移动逻辑：合并相同数字
        new_row = [i for i in row if i != 0]  # 移除零元素
        merged = []
        i = 0
        while i < len(new_row):
            if i + 1 < len(new_row) and new_row[i] == new_row[i + 1]:
                # 合并相邻相同元素，更新分数
                merged_val = new_row[i] * 2
                merged.append(merged_val)
                self.score += merged_val
                i += 2
            else:
                merged.append(new_row[i])
                i += 1
        # 补齐剩余位置为零
        return merged + [0] * (4 - len(merged))

    # 四个方向的移动实现（通过矩阵旋转简化处理）
    def _move_left(self, grid):
        return np.array([self._move(row) for row in grid])

    def _move_right(self, grid):
        reversed_grid = np.array([row[::-1] for row in grid])
        new_grid = np.array([self._move(row) for row in reversed_grid])
        return np.array([row[::-1] for row in new_grid])

    def _move_up(self, grid):
        rotated = np.rot90(grid, 1)  # 逆时针旋转90度
        new_grid = self._move_left(rotated)
        return np.rot90(new_grid, -1)  # 顺时针旋转90度复原

    def _move_down(self, grid):
        rotated = np.rot90(grid, -1)  # 顺时针旋转90度
        new_grid = self._move_left(rotated)
        return np.rot90(new_grid, 1)  # 逆时针旋转90度复原
