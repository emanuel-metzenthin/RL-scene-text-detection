import argparse
from typing import Text, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from text_localization_environment import TextLocEnv
from DQN import ImageDQN
from ICDAR_dataset import ICDARDataset
from actions import TriggerAction
from agent import Agent
from environment import Environment
from rl_dataset import RLDataset
from utils import get_closest_gt, display_image_tensor_with_bbox


class RLTraining(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.dqn = ImageDQN()
        self.target_dqn = ImageDQN()

        dataset = ICDARDataset(path='../data/ICDAR2013')

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=False,
            premasking=False,
            max_steps_per_image=50,
            bbox_scaling=0,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action='false',
        )

        self.agent = Agent(env)
        self.batch_reward = 0
        self.episode_reward = 0
        self.total_reward = 0
        self.hparams = hparams
        self.populate(self.hparams.warm_start_steps)
        self.episode = 0
        self.epsilon = 1

    def training_step(self, batch, idx):
        device = self.get_device(batch)

        reward, done = self.agent.play_step(self.dqn, epsilon=self.epsilon)
        self.episode_reward += reward
        self.epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                      self.episode + 1 / self.hparams.eps_last_episode)
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.episode += 1

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.log('total_reward', torch.tensor(self.total_reward).to(device), on_step=True, prog_bar=True)
        self.log('reward', torch.tensor(reward).to(device), on_step=True, prog_bar=True)
        self.log('episodes', self.episode, on_step=True, prog_bar=True)

        return {'loss': loss}

    def configure_optimizers(self):
        params_to_update = []
        for param in self.dqn.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return torch.optim.Adam(params_to_update, lr=0.01)

    def calculate_trigger_reward(self, iou, steps):
        return self.NU * iou - steps * self.P

    def train_dataloader(self) -> DataLoader:
        dataset = RLDataset(self.agent.replay_buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                sampler=None
                                )
        return dataloader

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.dqn, epsilon=1.0)

    def get_device(self, batch) -> str:
        return batch[0].device.index if self.on_gpu else 'cpu'

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.dqn(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_dqn(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = torch.tensor(next_state_values * self.hparams.gamma + rewards, dtype=torch.float32)

        return nn.MSELoss()(state_action_values, expected_state_action_values)

