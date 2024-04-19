import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_3D_chart(time_data, reward_data, action_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(time_data, reward_data, action_data, c='r', marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    ax.set_zlabel('Action')
    plt.show()
    ax.figure.savefig('QLearningChart.png')


class PlotCreator:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.rewards = np.zeros(num_episodes, dtype=int)
        self.episode_plot = None
        self.avg_plot = None
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.rewards[episode_index]

    def __setitem__(self, episode_index, episode_reward):
        self.rewards[episode_index] = episode_reward

    def create_plot(self):
        plt.style.use("ggplot")
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        # self.fig.canvas.set_window_title("Episode Reward History")
        self.ax.set_xlim(0, self.num_episodes + 5)
        self.ax.set_ylim(-210, -110)
        self.ax.set_title("Episode Reward History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Total Reward")
        self.episode_plot, = plt.plot([], [], linewidth=0.5, alpha=0.5,
                                    c="#1d619b", label="reward per episode")
        self.avg_plot, = plt.plot([], [], linewidth=3.0, alpha=0.8, c="#df3930",
                                label="average reward over the 200 last episodes")
        self.ax.legend(loc="upper left")

    def update_plot(self, episode_index):
        # update the episode plot
        x = range(episode_index)
        y = self.rewards[:episode_index]
        self.episode_plot.set_xdata(x)
        self.episode_plot.set_ydata(y)

        # update the average plot
        mean_kernel_size = 201
        rolling_mean_data = np.concatenate((np.full(mean_kernel_size, fill_value=-200),
                                            self.rewards[:episode_index]))
        rolling_mean_data = pd.Series(rolling_mean_data)

        rolling_means = rolling_mean_data.rolling(window=mean_kernel_size,
                                                min_periods=0).mean()[mean_kernel_size:]
        self.avg_plot.set_xdata(range(len(rolling_means)))
        self.avg_plot.set_ydata(rolling_means)

        plt.draw()
        plt.pause(0.0001)