import argparse
import os
from collections import deque
from typing import Text, Tuple

import neptune
import numpy as np
import torch
from torch import nn
from text_localization_environment import TextLocEnv
from tqdm import tqdm

from DQN import ImageDQN
from dataset.ICDAR_dataset import ICDARDataset
from dataset.sign_dataset import SignDataset
from agent import Agent
from evaluate import evaluate


def configure_optimizers(dqn, lr):
    params_to_update = []
    for param in dqn.parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return torch.optim.Adam(params_to_update, lr=lr)


def populate(agent, dqn, steps: int = 1000, device='cpu') -> None:
    """
    Carries out several random steps through the environment to initially fill
    up the replay buffer with experiences
    Args:
        steps: number of random steps to populate the buffer with
    """
    for i in range(steps):
        agent.play_step(dqn, epsilon=1, device=device)


def get_device(self, batch) -> str:
    return batch[0][0].device.index if self.on_gpu else 'cpu'


def dqn_mse_loss(batch: Tuple[torch.Tensor, torch.Tensor], dqn: nn.Module, target_dqn: nn.Module, hparams, device: Text) -> torch.Tensor:
    """
    Calculates the mse loss using a mini batch from the replay buffer
    Args:
        batch: current mini batch of replay data
    Returns:
        loss
    """
    states, actions, rewards, dones, next_states = batch
    states = torch.tensor(states[0], device=device), torch.tensor(states[1], device=device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device)
    next_states = torch.tensor(next_states[0], device=device), torch.tensor(next_states[1], device=device)

    state_action_values = dqn(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = target_dqn(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = torch.tensor(next_state_values * hparams.training.loss.gamma + rewards, dtype=torch.float32)

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def load_model_from_checkpoint(checkpoint, dqn, target_dqn):
    dqn.load_state_dict(torch.load(checkpoint))
    target_dqn.load_state_dict(torch.load(checkpoint))

    return dqn, target_dqn


def save_model(target_dqn, file_name, *kwargs):
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    save_dict = {
        "state_dict": target_dqn.state_dict(),
        **kwargs
    }
    torch.save(save_dict, './checkpoints/' + file_name)
    neptune.log_artifact('./checkpoints/' + file_name)


def load_dataset(dataset, split: Text = 'train'):
    if dataset == "icdar2013":
        return ICDARDataset(path='../data/ICDAR2013', split=split)
    elif dataset == "sign":
        return SignDataset(path='../data/600_3_signs_3_words', split=split)
    else:
        raise Exception(f"Dataset name {dataset} not supported.")


def train(hparams: argparse.Namespace):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset(hparams.dataset)

    env = TextLocEnv(
        dataset.images, dataset.gt,
        playout_episode=hparams.env.full_playout,
        premasking=hparams.env.premasking,
        max_steps_per_image=hparams.env.steps_per_image,
        bbox_scaling=0,
        bbox_transformer='base',
        ior_marker_type='cross',
        has_termination_action=hparams.env.termination,
    )

    dqn = ImageDQN(backbone=hparams.training.backbone, num_actions=len(env.action_set))
    target_dqn = ImageDQN(backbone=hparams.training.backbone, num_actions=len(env.action_set))

    if torch.cuda.device_count() > 1:
        dqn = nn.DataParallel(dqn)
        target_dqn = nn.DataParallel(target_dqn)

    if hparams.training.checkpoint:
        dqn, target_dqn = load_model_from_checkpoint(hparams.training.checkpoint, dqn, target_dqn)

    dqn.to(device)
    target_dqn.eval()
    target_dqn.to(device)
    optimizer = configure_optimizers(dqn, hparams.training.lr)

    agent = Agent(env)
    populate(agent, dqn)

    current_episode = 0
    training_step = 0
    running_reward = deque(maxlen=500)
    mean_reward = 0
    last_mean_reward = 0

    for current_epoch in range(hparams.training.epochs):
        # TODO run whole image dataset per epoch or predefined num. of steps
        render = True
        with tqdm(range(hparams.env.steps_per_epoch), unit='step') as tepoch:
            for current_step in tepoch:
                tepoch.set_description(f"Epoch {current_epoch}")

                epsilon = max(hparams.env.epsilon.end, hparams.env.epsilon.start -
                              current_episode / hparams.env.epsilon.last_episode)
                reward, done = agent.play_step(dqn, epsilon, device=device, render_on_trigger=False)

                if done:
                    current_episode += 1
                    running_reward.append(reward)
                    last_mean_reward = mean_reward
                    mean_reward = np.mean(running_reward)
                    render = False

                if current_step % hparams.training.update_interval == 0:
                    training_step += 1
                    experience_batch = agent.replay_buffer.sample(hparams.training.batch_size)

                    loss = dqn_mse_loss(experience_batch, dqn ,target_dqn, hparams, device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix({'loss': loss.item(), 'mean_episode_reward': mean_reward, 'epsilon': epsilon})
                    neptune.log_metric('loss', loss)
                    neptune.log_metric('mean_episode_reward', mean_reward)
                    neptune.log_metric('epsilon', epsilon)

                    if training_step % hparams.training.sync_rate == 0:
                        target_dqn.load_state_dict(dqn.state_dict())

        if current_epoch > 0 and current_epoch % hparams.validation.every == 0:
            avg_iou = evaluate(hparams, agent, target_dqn, device)
            neptune.log_metric('avg_iou', avg_iou)

        # TODO also save epoch etc., log model to neptune
        if mean_reward > last_mean_reward:
            file_name = f'{hparams.neptune.run_name}_best.pt'
            save_model(target_dqn, file_name,
                       mean_reward=mean_reward, loss=loss, current_epoch=current_epoch, epsilon=epsilon)
