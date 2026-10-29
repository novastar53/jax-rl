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



class LiveMCConvergence:
    def __init__(self, size=4, interval=0.01):
        self.size = size
        self.interval = interval

        plt.ion()

        self.fig, (self.ax_v, self.ax_error) = plt.subplots(
            1, 2, figsize=(13, 5.5)
        )

        self.im = self.ax_v.imshow(
            np.zeros((size, size)),
            cmap="viridis",
            vmin=-25,
            vmax=0,
        )

        self.texts = [
            [
                self.ax_v.text(
                    j, i, "0.0",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                )
                for j in range(size)
            ]
            for i in range(size)
        ]

        self.ax_v.set_xticks(range(size))
        self.ax_v.set_yticks(range(size))
        self.ax_v.set_title("Monte Carlo Values")

        self.fig.colorbar(self.im, ax=self.ax_v, shrink=0.8)

        (self.error_line,) = self.ax_error.plot(
            [], [], linewidth=2
        )

        self.ax_error.set_xlabel("Episodes")
        self.ax_error.set_ylabel("RMSE")
        self.ax_error.set_title("MC Prediction Error")
        self.ax_error.grid(alpha=0.3)

        self.episodes = []
        self.errors = []

        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, V, episode, rmse):
        grid = np.asarray(V).reshape(self.size, self.size)

        self.im.set_data(grid)
        self.ax_v.set_title(
            f"Monte Carlo Values — Episode {episode}"
        )

        for i in range(self.size):
            for j in range(self.size):
                self.texts[i][j].set_text(
                    f"{grid[i, j]:.1f}"
                )

        self.episodes.append(episode)
        self.errors.append(float(rmse))

        self.error_line.set_data(
            self.episodes, self.errors
        )

        self.ax_error.relim()
        self.ax_error.autoscale_view()

        self.fig.canvas.draw_idle()
        plt.pause(self.interval)

    def close(self):
        plt.ioff()
        plt.show()



class LiveMCComparison:
    def __init__(self, size=4, interval=0.01):
        self.size = size
        self.interval = interval

        plt.ion()

        self.fig, axes = plt.subplots(
            1, 3,
            figsize=(17, 5.5),
            gridspec_kw={"width_ratios": [1, 1, 1.5]},
        )

        self.ax_first, self.ax_every, self.ax_error = axes

        self.images = []
        self.texts = []

        for ax, title in [
            (self.ax_first, "First-Visit MC"),
            (self.ax_every, "Every-Visit MC"),
        ]:
            image = ax.imshow(
                np.zeros((size, size)),
                cmap="viridis",
                vmin=-25,
                vmax=0,
            )

            ax.set_title(title)
            ax.set_xticks(range(size))
            ax.set_yticks(range(size))

            labels = [
                [
                    ax.text(
                        j, i, "0.0",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=11,
                    )
                    for j in range(size)
                ]
                for i in range(size)
            ]

            self.images.append(image)
            self.texts.append(labels)

        self.fig.colorbar(
            self.images[1],
            ax=[self.ax_first, self.ax_every],
            shrink=0.75,
        )

        (self.first_line,) = self.ax_error.plot(
            [], [], label="First-visit MC", linewidth=2
        )
        (self.every_line,) = self.ax_error.plot(
            [], [], label="Every-visit MC", linewidth=2
        )

        self.ax_error.set_title("Convergence to DP Values")
        self.ax_error.set_xlabel("Episodes")
        self.ax_error.set_ylabel("RMSE")
        self.ax_error.grid(alpha=0.3)
        self.ax_error.legend()

        self.episodes = []
        self.first_errors = []
        self.every_errors = []

        self.fig.subplots_adjust(
            left=0.05, right=0.97,
            bottom=0.13, top=0.88,
            wspace=0.35,
        )

        plt.show(block=False)

    def update(
        self,
        episode,
        V_first,
        V_every,
        rmse_first,
        rmse_every,
    ):
        for index, values in enumerate([V_first, V_every]):
            grid = np.asarray(values).reshape(
                self.size, self.size
            )

            self.images[index].set_data(grid)

            for i in range(self.size):
                for j in range(self.size):
                    self.texts[index][i][j].set_text(
                        f"{grid[i, j]:.1f}"
                    )

        self.ax_first.set_title(
            f"First-Visit MC — Episode {episode}"
        )
        self.ax_every.set_title(
            f"Every-Visit MC — Episode {episode}"
        )

        self.episodes.append(episode)
        self.first_errors.append(float(rmse_first))
        self.every_errors.append(float(rmse_every))

        self.first_line.set_data(
            self.episodes, self.first_errors
        )
        self.every_line.set_data(
            self.episodes, self.every_errors
        )

        self.ax_error.relim()
        self.ax_error.autoscale_view()

        self.fig.canvas.draw_idle()
        plt.pause(self.interval)

    def close(self):
        plt.ioff()
        plt.show()
