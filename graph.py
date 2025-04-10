import matplotlib.pyplot as plt
import numpy as np

class ShowGraph:
    def save_graph(self,reward_per_episode, epsilon_history,file):
        fig = plt.figure(1)

        mean_reward = np.zeros(len(reward_per_episode))

        for i in range(len(mean_reward)):
            mean_reward[i] = np.mean(reward_per_episode[max(0,i-99):i+1])
        plt.subplot(121)
        plt.ylabel('Mean Reward')
        plt.plot(mean_reward)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(file)
        plt.close()


