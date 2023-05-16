import time
import torch
import numpy as np
from catch import CatchEnv
from dqn_pytorch import DQN_Agent

def test(env, agent, device, run):
    rewards = []

    for _ in range(10):
        current_state = np.array([np.transpose(np.array(env.reset()), (2, 0, 1))])
        done = False

        while not done:
            current_state = torch.Tensor(current_state).to(device)
            action = agent.policy_net(current_state).max(1)[1].view(1, 1).item()
            next_state, reward, done = env.step(action)
            next_state = np.array([np.transpose(np.array(next_state), (2, 0, 1))])
            if done:
                rewards.append(reward)
                break
            current_state = next_state
    
    with open('run_' + str(run+1) + '.txt', 'a') as f:
        f.write(str(np.mean(rewards))+'\n')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CatchEnv()
    BATCH_SIZE = 32

    state_shape = env.state_shape()
    num_actions = env.get_num_actions()
    
    for run in range(5):
        agent = DQN_Agent(num_actions, state_shape, device)

        num_transitions = 0
        start_time = time.time()
        for e in range(4000):
            current_state = np.array([np.transpose(np.array(env.reset()), (2, 0, 1))])
            done = False
            while not done:
                num_transitions += 1
                action = agent.select_action(current_state)
                next_state, reward, done = env.step(action)
                next_state = np.array([np.transpose(np.array(next_state), (2, 0, 1))])
                agent.store_transition(current_state, action, reward, next_state, done)

                if num_transitions >= BATCH_SIZE:
                    agent.train(BATCH_SIZE)
                    if num_transitions % agent.target_update_rate == 0:
                        agent.update_target()

                current_state = next_state

            if (e+1) % 10 == 0:
                print(f"Elapsed time: {time.time() - start_time}s")
                print(f"Epsilon: {agent.epsilon}")
                test(env, agent, device, run)
