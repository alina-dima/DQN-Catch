import random
import numpy as np
import torch
from collections import deque
from dqn_net import DQNet


class DQN_Agent():
    def __init__(self, num_actions, state_shape, device):
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.device = device

        self.replay_capacity = 4000
        self.replay_memory = deque(maxlen=self.replay_capacity)

        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.000075
        self.gamma = 0.99
        self.target_update_rate = 800

        self.policy_net = DQNet(num_actions, state_shape[0]).to(device)
        self.target_net = DQNet(num_actions, state_shape[0]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.00025, momentum=0.95, eps=0.01)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions-1)

        state = torch.Tensor(state).to(self.device)
        return self.policy_net(state).max(1)[1].view(1, 1).item()

    def store_transition(self, state, action, reward, next_state, terminal):
        if len(self.replay_memory) == self.replay_capacity:
            self.replay_memory.pop()

        self.replay_memory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminal": terminal
        })

    def sample_replay(self, minibatch_size):
        minibatch = random.sample(self.replay_memory, minibatch_size)
        states = np.array([transition['state'][0] for transition in minibatch])
        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor([transition['action'] for transition in minibatch]).to(self.device)
        rewards = torch.Tensor([transition['reward'] for transition in minibatch]).to(self.device)
        next_states = np.array([transition['next_state'][0] for transition in minibatch])
        next_states = torch.Tensor(next_states).to(self.device)
        terminals = torch.Tensor([transition['terminal'] for transition in minibatch]).to(self.device)

        return states, actions, rewards, next_states, terminals

    def decrease_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon-self.epsilon_decay)

    def train(self, minibatch_size):
        states, actions, rewards, next_states, terminals = self.sample_replay(minibatch_size)
        q_values = self.policy_net(states).gather(1, actions.type(torch.int64).unsqueeze(-1)).squeeze(-1)

        next_max_qs = self.target_net(next_states).max(1)[0]
        targets = rewards + (1-terminals) * self.gamma * next_max_qs

        loss = torch.nn.SmoothL1Loss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decrease_epsilon()
