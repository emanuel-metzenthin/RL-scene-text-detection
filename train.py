import argparse
from typing import Text, List, Tuple, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from text_localization_environment import TextLocEnv
from DQN import ImageDQN
from ICDAR_dataset import ICDARDataset
from agent import Agent
from rl_dataset import RLDataset
from utils import get_closest_gt, display_image_tensor_with_bbox


class RLTraining(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams

        dataset = ICDARDataset(path='../data/ICDAR2013')
        self.test_dataset = ICDARDataset(path='../data/ICDAR2013', split='test')

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=False,
            premasking=False,
            max_steps_per_image=self.hparams.steps_per_image,
            bbox_scaling=0,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=False,
        )

        self.test_env = TextLocEnv(
            self.test_dataset.images, self.test_dataset.gt,
            playout_episode=False,
            premasking=False,
            max_steps_per_image=self.hparams.steps_per_image,
            bbox_scaling=0,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=False,
            mode='test'
        )

        self.agent = Agent(env)

        self.dqn = ImageDQN(num_actions=len(env.action_set))
        self.target_dqn = ImageDQN(num_actions=len(env.action_set))

        self.episode = 0
        self.batch_reward = 0
        self.episode_reward = 0
        self.total_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def training_step(self, batch, idx):
        device = self.get_device(batch)

        loss = self.dqn_mse_loss(batch, device)

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        return {'loss': loss}

    def training_epoch_start(self) -> None:
        if self.current_epoch == 0:
            return
        self.populate(self.hparams.warm_start_steps)
        self.log('total_reward', self.total_reward, on_epoch=True, prog_bar=True)
        self.log('episodes', self.episode, on_epoch=True, prog_bar=True)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.current_epoch % self.hparams.evaluate_every == 1:
            avg_iou = self.evaluate()

            self.log('avg_iou', avg_iou)

    def evaluate(self):
        self.target_dqn.eval()

        avg_iou = 0
        num_images = min(self.hparams.num_epoch_eval_images, len(self.test_env.image_paths)) \
            if self.hparams.num_epoch_eval_images \
            else len(self.test_env.image_paths)

        for image_idx in range(num_images):
            self.test_env.reset(image_index=image_idx)
            done = False

            while not done:
                action = self.agent.choose_action(self.target_dqn, 0, device=self.device)
                state, reward, done, _ = self.test_env.step(action)

            if len(self.test_env.episode_trigger_ious) > 0:
                avg_iou += np.mean(self.test_env.episode_trigger_ious)

            if self.hparams.neptune_key and image_idx < 5:
                self.logger.experiment.log_image(f'sample_image_{image_idx}', self.test_env.render(return_as_file=True))

        return avg_iou / num_images

    def configure_optimizers(self):
        params_to_update = []
        for param in self.dqn.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return torch.optim.Adam(params_to_update, lr=0.01)

    def train_dataloader(self) -> DataLoader:
        dataset = RLDataset(self.agent.replay_buffer, min(1_000, self.hparams.replay_buffer_sample_size))
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
            epsilon = max(self.hparams.eps_end, self.hparams.eps_start -
                          self.episode + 1 / self.hparams.eps_last_episode)
            reward, done = self.agent.play_step(self.dqn, epsilon=epsilon)

            self.episode_reward += reward

            if done:
                self.total_reward = self.episode_reward
                self.episode_reward = 0
                self.episode += 1

    def get_device(self, batch) -> str:
        return batch[0][0].device.index if self.on_gpu else 'cpu'

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor], device: Text) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        states = torch.tensor(states[0]).to(device), torch.tensor(states[1]).to(device)

        state_action_values = self.dqn(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_dqn(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = torch.tensor(next_state_values * self.hparams.gamma + rewards, dtype=torch.float32)

        return nn.MSELoss()(state_action_values, expected_state_action_values)
