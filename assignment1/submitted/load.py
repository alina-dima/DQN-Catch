import numpy as np
import matplotlib.pyplot as plt

X = np.load('group_27_catch_rewards_1.npy', mmap_mode='r')

ticks = [(i+1)*10 for i in range(300)]
plt.plot(ticks, X)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.savefig('group_27_catch_rewards_1.png')
plt.clf()

X = np.load('group_27_catch_rewards_2.npy', mmap_mode='r')

ticks = [(i+1)*10 for i in range(300)]
plt.plot(ticks, X)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.savefig('group_27_catch_rewards_2.png')
plt.clf()

X = np.load('group_27_catch_rewards_3.npy', mmap_mode='r')

ticks = [(i+1)*10 for i in range(300)]
plt.plot(ticks, X)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.savefig('group_27_catch_rewards_3.png')
plt.clf()

X = np.load('group_27_catch_rewards_4.npy', mmap_mode='r')

ticks = [(i+1)*10 for i in range(300)]
plt.plot(ticks, X)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.savefig('group_27_catch_rewards_4.png')
plt.clf()

X = np.load('group_27_catch_rewards_5.npy', mmap_mode='r')

ticks = [(i+1)*10 for i in range(300)]
plt.plot(ticks, X)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.savefig('group_27_catch_rewards_5.png')
plt.clf()