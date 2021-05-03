import random
import uuid
from collections import deque
from copy import deepcopy
from typing import Text

import neptune
import numpy as np
import ray
import torch
from torch import nn
from replay_buffer import Experience, ReplayBuffer


@ray.remote
class Actor:
    def __init__(self, dqn, target_dqn, env, replay_buffer_handle, learner_handle, param_server_handle, logger, cfg, actor_id):
        self.actor_id = actor_id
        self.env = env
        self.total_reward = 0
        self.state = None
        self.local_buffer = []
        self.remote_buffer = replay_buffer_handle
        self.action_freq = torch.from_numpy(np.zeros(self.env.action_space.n))
        self.device = torch.device("cpu")
        self.total_steps = 0
        self.dqn = dqn.to(self.device)
        self.target_dqn = target_dqn.to(self.device)
        self.cfg = cfg
        self.learner = learner_handle
        self.param_server = param_server_handle
        self.logger = logger
        self.last_episode_rewards = deque(maxlen=10)
        self.current_episode_reward = 0
        self.reset()

    def choose_action(self, dqn: nn.Module, epsilon: float, upper_confidence_bound=False, time_step=None):
        if not upper_confidence_bound and np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(self.state[0]).unsqueeze(0).to(self.device), torch.tensor(self.state[1]).unsqueeze(0).to(self.device)
            dqn.eval()
            with torch.no_grad():
                q_values = dqn(state).squeeze(0)
            dqn.train()

            if upper_confidence_bound:
                _, action = torch.max(q_values + 1.5 * torch.sqrt(self.action_freq * (1 / np.log(time_step))), dim=0)
            else:
                _, action = torch.max(q_values, dim=0)
            action = int(action.item())

        self.action_freq[action] += 1

        return action

    def reset(self, image_index=None) -> None:
        """ Resets the environment and updates the state"""
        self.state = self.env.reset(image_index)
        self.current_episode_reward = 0

    def play_step(self, dqn: nn.Module, epsilon=0.0, render_on_trigger=False, upper_confidence_bound=False, time_step=None):
        action = self.choose_action(dqn, epsilon, upper_confidence_bound, time_step)

        if self.env.is_trigger(action) and render_on_trigger:
            neptune.log_image(f'sample_image_{str(uuid.uuid4())[:8]}', self.env.render(return_as_file=True, include_true_bboxes=True))

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.local_buffer.append(exp)

        self.state = new_state

        if done:
            self.current_episode_reward += reward
            self.last_episode_rewards.append(self.current_episode_reward)
            if self.logger:
                self.logger.log.remote('train/mean_episode_reward', np.mean(self.last_episode_rewards))
                # self.neptune_run['train/epsilon'].log(epsilon)
            self.reset()

        return reward, done

    def collect_experience(self):
        self.local_buffer = []

        while len(self.local_buffer) < 100:
            epsilon = max(self.cfg.env.epsilon.end, self.cfg.env.epsilon.start -
                          (self.total_steps / self.cfg.env.epsilon.decay_steps) * (self.cfg.env.epsilon.start - self.cfg.env.epsilon.end))
            self.play_step(self.dqn, epsilon=epsilon, render_on_trigger=False, upper_confidence_bound=False, time_step=None)

    def send_off_replay_data(self):
        self.remote_buffer.append.remote(self.local_buffer)
        # print(f"Actor {self.actor_id}: sending off replay data")

    def receive_new_parameters(self):
        params_ref = ray.wait([self.param_server.get_current_parameters.remote()])
        print(f"received parrams {params_ref}")
        if params_ref:
            new_params_dqn, new_params_target_dqn = ray.get(params_ref)
            # print(f"Actor {self.actor_id}: received new params")

            for param, new_param in zip(self.dqn.parameters(), new_params_dqn):
                new_param = torch.FloatTensor(new_param).to(self.device)
                param.data.copy_(new_param)

            for param, new_param in zip(self.target_dqn.parameters(), new_params_target_dqn):
                new_param = torch.FloatTensor(new_param).to(self.device)
                param.data.copy_(new_param)

    def run(self):
        while True:
            self.collect_experience()
            self.send_off_replay_data()
            self.receive_new_parameters()



