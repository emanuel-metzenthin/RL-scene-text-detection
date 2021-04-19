import numpy as np
from text_localization_environment import TextLocEnv
from tqdm import tqdm

from dataset.ICDAR_dataset import ICDARDataset
from agent import Agent

test_dataset = ICDARDataset(path='../data/ICDAR2013', split='test')


def f_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def evaluate(hparams, dqn, device='cpu'):
    avg_iou = 0

    test_env = TextLocEnv(
        test_dataset.images, test_dataset.gt,
        playout_episode=hparams.env.full_playout,
        premasking=hparams.env.premasking,
        max_steps_per_image=200,
        bbox_scaling=0,
        bbox_transformer='base',
        ior_marker_type='cross',
        has_termination_action=hparams.env.termination,
        mode='test'
    )
    agent = Agent(test_env)
    num_images = len(test_dataset.images)

    tqdm.write('Evaluating...')
    with tqdm(range(num_images)) as timages:
        for image_idx in timages:
            test_env.reset(image_index=image_idx)
            done = False

            while not done:
                action = agent.choose_action(dqn, 0, device=device)
                state, reward, done, _ = test_env.step(action)

            if len(test_env.episode_trigger_ious) > 0:
                avg_iou += np.mean(test_env.episode_trigger_ious)

            timages.set_postfix({'avg_iou': avg_iou / num_images})

    return avg_iou / num_images