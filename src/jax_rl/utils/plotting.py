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


class LivePolicyIteration:
    ARROWS = ["↑", "→", "↓", "←"]

    def __init__(self, size=4, interval=0.6):
        self.size = size
        self.interval = interval

        plt.ion()

        self.fig, (self.ax_v, self.ax_pi) = plt.subplots(
            1, 2, figsize=(12, 6)
        )

        # Left: state values
        self.im = self.ax_v.imshow(
            np.zeros((size, size)),
            cmap="viridis",
            vmin=-25,
            vmax=0,
        )

        self.value_texts = [
            [
                self.ax_v.text(
                    j, i, "0.0",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=13,
                )
                for j in range(size)
            ]
            for i in range(size)
        ]

        # Right: policy directions
        self.ax_pi.imshow(
            np.zeros((size, size)),
            cmap="Greys",
            vmin=0,
            vmax=1,
        )

        self.policy_texts = [
            [
                self.ax_pi.text(
                    j, i, "",
                    ha="center",
                    va="center",
                    fontsize=17,
                )
                for j in range(size)
            ]
            for i in range(size)
        ]

        for ax in (self.ax_v, self.ax_pi):
            ax.set_xticks(range(size))
            ax.set_yticks(range(size))
            ax.set_xticks(
                np.arange(-0.5, size, 1), minor=True
            )
            ax.set_yticks(
                np.arange(-0.5, size, 1), minor=True
            )
            ax.grid(
                which="minor", color="gray",
                linewidth=1
            )
            ax.tick_params(which="minor", length=0)

        self.fig.colorbar(
            self.im, ax=self.ax_v, shrink=0.8
        )

        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, V, policy, iteration, phase=""):
        values = np.asarray(V).reshape(
            self.size, self.size
        )
        probabilities = np.asarray(policy)

        self.im.set_data(values)

        self.ax_v.set_title(
            f"State Values — Iteration {iteration}"
        )
        self.ax_pi.set_title(
            f"Policy — {phase}"
        )

        for i in range(self.size):
            for j in range(self.size):
                state = i * self.size + j

                self.value_texts[i][j].set_text(
                    f"{values[i, j]:.2f}"
                )

                if state in (0, self.size**2 - 1):
                    label = "■"
                else:
                    actions = np.flatnonzero(
                        probabilities[state] > 0
                    )
                    label = "".join(
                        self.ARROWS[a] for a in actions
                    )

                self.policy_texts[i][j].set_text(label)

        self.fig.canvas.draw_idle()
        plt.pause(self.interval)

    def close(self):
        plt.ioff()
        plt.show()
