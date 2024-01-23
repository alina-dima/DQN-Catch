import numpy as np
import matplotlib.pyplot as plt

# Plot average rewards of the 5 runs
for run in range(5):
    X = np.load('group_27_catch_rewards_' + str(run+1) + '.npy', mmap_mode='r')
    ticks = [(i+1)*10 for i in range(300)]
    plt.plot(ticks, X)
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.savefig('group_27_catch_rewards_' + str(run+1))
    plt.clf()
X = np.load('group_27_catch_rewards_1.npy', mmap_mode='r')
