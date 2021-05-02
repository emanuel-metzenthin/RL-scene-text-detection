import argparse
import os
from collections import deque
from typing import Text, Tuple, Dict

import numpy as np
import torch
from neptune.new.types import File
from text_localization_environment import TextLocEnv
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import neptune
from DQN import ImageDQN
from dataset.ICDAR_dataset import ICDARDataset
from dataset.sign_dataset import SignDataset
import plotly.express as px


# from evaluate import evaluate
from dataset.simple_dataset import SimpleDataset


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





def load_model_from_checkpoint(checkpoint, dqn, target_dqn):
    checkpoint_dict = torch.load(checkpoint)
    dqn.load_state_dict(checkpoint_dict['state_dict'])
    target_dqn.load_state_dict(checkpoint_dict['state_dict'])

    return dqn, target_dqn, checkpoint_dict


def save_model(target_dqn, file_name, run, **kwargs):
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    save_dict = {
        "state_dict": target_dqn.state_dict(),
        **kwargs
    }
    torch.save(save_dict, './checkpoints/' + file_name)
    if run:
        run['model_checkpoint'].upload('./checkpoints/' + file_name)


def train(hparams: argparse.Namespace, run: Dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = GradScaler()
    dataset = load_dataset(hparams.dataset)

    current_episode = 0
    training_step = 0
    total_steps = 0
    running_reward = deque(maxlen=10)
    episode_rewards = []
    mean_reward = 0
    last_mean_reward = 0

    env = TextLocEnv(
        dataset.images, dataset.gt,
        playout_episode=hparams.env.full_playout,
        premasking=hparams.env.premasking,
        max_steps_per_image=hparams.env.steps_per_image,
        bbox_scaling=0,
        bbox_transformer='base',
        ior_marker_type='cross',
        has_termination_action=hparams.env.termination,
        has_intermediate_reward=hparams.env.intermediate_reward
    )

    dqn = ImageDQN(backbone=hparams.training.backbone, num_actions=len(env.action_set))
    target_dqn = ImageDQN(backbone=hparams.training.backbone, num_actions=len(env.action_set))

    if torch.cuda.device_count() > 1:
        dqn = nn.DataParallel(dqn)
        target_dqn = nn.DataParallel(target_dqn)

    if hparams.training.checkpoint:
        dqn, target_dqn, checkpoint_dict = load_model_from_checkpoint(hparams.training.checkpoint, dqn, target_dqn)
        current_episode = checkpoint_dict['current_epoch']
        mean_reward = checkpoint_dict['mean_reward']
        hparams.env.epsilon.start = checkpoint_dict['epsilon']

    dqn.to(device)
    target_dqn.eval()
    target_dqn.to(device)
    optimizer = configure_optimizers(dqn, hparams.training.lr)

    agent = Agent(env, hparams.env.replay_buffer.size)
    populate(agent, dqn, steps=hparams.env.replay_buffer.warm_up_steps)

    for current_epoch in range(hparams.training.epochs):
        # TODO run whole image dataset per epoch or predefined num. of steps
        render = True
        with tqdm(range(hparams.env.steps_per_epoch), unit='step') as tepoch:
            for current_step in tepoch:
                tepoch.set_description(f"Epoch {current_epoch}")
                total_steps += 1
                epsilon = max(hparams.env.epsilon.end, hparams.env.epsilon.start -
                              total_steps / hparams.env.epsilon.decay_steps)
                reward, done = agent.play_step(dqn, epsilon, device=device, render_on_trigger=False,
                                               upper_confidence_bound=hparams.env.upper_confidence_bound, time_step=total_steps)

                episode_rewards.append(reward)
                if done:
                    current_episode += 1
                    total_reward = sum(episode_rewards)
                    episode_rewards = []
                    running_reward.append(total_reward)
                    last_mean_reward = mean_reward
                    mean_reward = np.mean(running_reward)
                    render = False

                if current_step % hparams.training.update_interval == 0:
                    training_step += 1
                    experience_batch = agent.replay_buffer.sample(hparams.training.batch_size)

                    optimizer.zero_grad()
                    with autocast():
                        loss = dqn_mse_loss(experience_batch, dqn, target_dqn, hparams, device)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if training_step % hparams.training.sync_rate == 0:
                        target_dqn.load_state_dict(dqn.state_dict())

                tepoch.set_postfix({'loss': loss.item(), 'mean_episode_reward': mean_reward, 'epsilon': epsilon})
                if run:
                    run['train/loss'].log(loss)
                    run['train/mean_episode_reward'].log(mean_reward)
                    run['train/epsilon'].log(epsilon)

        # if current_epoch > 0 and current_epoch % hparams.validation.every == 0:
        #     avg_iou = evaluate(hparams, agent, target_dqn, device)
        #     run['val/avg_iou'].log(avg_iou)

        if mean_reward > last_mean_reward:
            file_name = f'{hparams.neptune.run_name}_best.pt'
            save_model(target_dqn, file_name, run,
                       mean_reward=mean_reward, loss=loss, current_epoch=current_epoch, epsilon=epsilon)

        if run:
            action_barchart = px.bar(x=range(env.action_space.n), y=agent.action_freq.detach())
            run['visuals/action_frequency'].upload(File.as_html(action_barchart))
