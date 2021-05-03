from asyncio import sleep
from typing import Tuple

import ray
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, dqn: nn.Module, target_dqn: nn.Module, logger, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.apex.learner_device)
        self.dqn = dqn.to(self.device)
        self.target_dqn = target_dqn.to(self.device)
        self.current_batch = None
        self.current_training_step = 0
        self.current_params_id = None
        self.scaler = GradScaler()
        self.optimizer = self.configure_optimizers()
        self.logger = logger

    def configure_optimizers(self):
        params_to_update = []
        for param in self.dqn.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        return torch.optim.Adam(params_to_update, lr=self.cfg.training.lr)

    def dqn_loss(self, batch: Tuple[torch.Tensor, torch.Tensor], use_mse=False) -> torch.Tensor:
        """
        Calculates the huber/mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
            use_mse: Use MSE loss instead ob huber loss
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        states = torch.from_numpy(np.array(states[0])).to(self.device), torch.from_numpy(np.array(states[1])).to(self.device)
        actions = torch.from_numpy(np.array(actions)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).to(self.device)
        # np.array calls as performance fix: https://github.com/pytorch/pytorch/issues/13918
        next_states = torch.from_numpy(np.array(next_states[0])).to(self.device), torch.from_numpy(np.array(next_states[1])).to(self.device)

        state_action_values = self.dqn(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_dqn(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = torch.tensor(next_state_values * self.cfg.training.loss.gamma + rewards, dtype=torch.float32)

        if use_mse:
            return nn.MSELoss()(state_action_values, expected_state_action_values)
        else:
            return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

    def training_step(self):
        self.current_training_step += 1
        self.optimizer.zero_grad()
        with autocast():
            loss = self.dqn_loss(self.current_batch)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.current_training_step % self.cfg.training.sync_rate == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        if self.logger:
            self.logger.log.remote('train/loss', loss.detach().to("cpu"))

        print(f"Learner: step {self.current_training_step} finished")

    def receive_batch(self):
        self.current_batch = ray.get(ray.get_actor("replay_buffer").get_next_batch.remote())

        while not self.current_batch:
            sleep(1)
            self.current_batch = ray.get(ray.get_actor("replay_buffer").get_next_batch.remote())

    def publish_parameters(self):
        object_ref = ray.put(self.dqn.parameters().to("cpu"))
        print("want to publish")
        ray.get_actor("param_server").publish_parameters.remote(object_ref)
        print("have published")

    def run(self):
        while True:
            self.receive_batch()
            self.training_step()
            if self.current_training_step % self.cfg.apex.publish_params_interval == 0:
                self.publish_parameters()
