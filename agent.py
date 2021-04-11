import random
from typing import Text

import numpy as np
import torch
from torch import nn
from replay_buffer import Experience, ReplayBuffer


class Agent:
    def __init__(self, env):
        self.env = env
        self.replay_buffer = ReplayBuffer(1_000)
        self.total_reward = 0
        self.reset()

    def choose_action(self, dqn: nn.Module, epsilon: float, device: Text):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(self.state[0]).unsqueeze(0).to(device), torch.tensor(self.state[1]).unsqueeze(0).to(device)
            dqn.eval()
            with torch.no_grad():
                q_values = dqn(state).squeeze(0)
            dqn.train()
            _, action = torch.max(q_values, dim=0)
            action = int(action.item())

        return action

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def play_step(self, dqn: nn.Module, epsilon=0.0, device: Text='cpu'):
        action = self.choose_action(dqn, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state

        if done:
            self.reset()

        return reward, done




