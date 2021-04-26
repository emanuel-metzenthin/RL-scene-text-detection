import random
import uuid
from typing import Text

import neptune
import numpy as np
import torch
from torch import nn
from replay_buffer import Experience, ReplayBuffer


class Agent:
    def __init__(self, env, buffer_capacity=1_000):
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_capacity)
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

    def reset(self, image_index=None) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset(image_index)

    def play_step(self, dqn: nn.Module, epsilon=0.0, device: Text='cpu', render_on_trigger=False):
        action = self.choose_action(dqn, epsilon, device)

        if self.env.is_trigger(action) and render_on_trigger:
            neptune.log_image(f'sample_image_{str(uuid.uuid4())[:8]}', self.env.render(return_as_file=True, include_true_bboxes=True))

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state

        if done:
            self.reset()

        return reward, done




