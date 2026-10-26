import numpy as np
import matplotlib.pyplot as plt


class LiveValueGrid:
    def __init__(self, size=4, interval=0.1):
        self.size = size
        self.interval = interval

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.im = self.ax.imshow(
            np.zeros((size, size)),
            cmap="viridis",
            vmin=-25,
            vmax=0,
        )

        self.ax.set_xticks(range(size))
        self.ax.set_yticks(range(size))

        self.texts = [
            [
                self.ax.text(
                    j, i, "0.0",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14,
                )
                for j in range(size)
            ]
            for i in range(size)
        ]

        self.fig.colorbar(self.im, ax=self.ax)
        self.ax.set_title("Policy Evaluation — Sweep 0")

        plt.show(block=False)

    def update(self, V, iteration):
        grid = np.asarray(V).reshape(self.size, self.size)

        self.im.set_data(grid)
        self.ax.set_title(
            f"Policy Evaluation — Sweep {iteration}"
        )

        for i in range(self.size):
            for j in range(self.size):
                self.texts[i][j].set_text(
                    f"{grid[i, j]:.2f}"
                )

        self.fig.canvas.draw_idle()
        plt.pause(self.interval)

    def close(self):
        plt.ioff()
        plt.show()