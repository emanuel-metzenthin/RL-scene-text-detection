import os
import time

import numpy as np
import torch
from text_localization_environment import TextLocEnv
from tqdm import tqdm

from DQN import ImageDQN
from dataset.ICDAR_dataset import ICDARDataset
from agent import Agent
from dataset.sign_dataset import SignDataset

test_dataset = SignDataset(path='../data/600_3_signs_3_words', split='validation')


def evaluate(dqn, env, device='cpu'):
    avg_iou = 0

    agent = Agent(env)
    num_images = 1000 # len(test_dataset.images)

    with tqdm(range(num_images)) as timages:
        for image_idx in timages:
            if not os.path.exists('./results'):
                os.makedirs('./results')
            test_file = open(f'./results/res_img_{image_idx}.txt', 'w+')
            agent.reset(image_index=image_idx)
            done = False

            while not done:
                action = agent.choose_action(dqn, 0, device)
                # do step in the environment
                new_state, _, done, _ = env.step(action)
                agent.state = new_state
                # env.render(mode='interactive')
                # time.sleep(0.1)

            for bbox in env.episode_pred_bboxes:
                test_file.write(f"{','.join(map(str, map(int, bbox)))}\n")

            test_file.close()

    return avg_iou / num_images


if __name__ == '__main__':
    test_env = TextLocEnv(
        test_dataset.images, test_dataset.gt,
        playout_episode=True,
        premasking=False,
        max_steps_per_image=200,
        bbox_scaling=0,
        bbox_transformer='base',
        ior_marker_type='cross',
        has_termination_action=False,
        mode='test'
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dqn = ImageDQN(num_actions=len(test_env.action_set)).to(device)
    checkpoint = torch.load('./checkpoints/sign_intermediate_best.pt', map_location=device)
    dqn.load_state_dict(checkpoint['state_dict'])

    evaluate(dqn, test_env, device)

    # ious = []
    # for i in range(1000):
    #     gt = []
    #     for line in open(f'sign_gt/gt_img_{i}.txt').readlines():
    #         gt.append(list(map(int, line.split(',')[:4])))
    #
    #     res = []
    #     for line in open(f'sign_res/res_img_{i}.txt').readlines():
    #         res.append(list(map(int, line.split(',')[:4])))
    #
    #     test_env.episode_true_bboxes = gt
    #     for b in res:
    #         print('..')
    #         test_env.bbox = b
    #         ious.append(test_env.compute_best_iou())
    # print(ious)
    # print(np.mean(ious))