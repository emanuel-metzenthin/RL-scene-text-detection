import os
import time
import zipfile

import numpy as np
import ray
import torch
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.tune import register_env
from ray.tune.utils import merge_dicts
from text_localization_environment import TextLocEnv
from tqdm import tqdm
from env_factory import EnvFactory


def evaluate(agent, env):
    num_images = 1000 # len(test_dataset.images)

    with tqdm(range(num_images)) as timages:
        _DUMMY_AGENT_ID = "agent0"

        print("Creating .zip file")
        zipf = zipfile.ZipFile('./results/res.zip', 'w', zipfile.ZIP_DEFLATED)

        for image_idx in timages:
            if not os.path.exists('./results'):
                os.makedirs('./results')
            test_file = open(f'./results/res_img_{image_idx}.txt', 'w+')
            obs = {_DUMMY_AGENT_ID: env.reset(image_index=image_idx)}
            done = False

            while not done:
                action = agent.compute_actions(obs)
                # do step in the environment
                obs[_DUMMY_AGENT_ID], r, done, _ = env.step(action[_DUMMY_AGENT_ID])
                env.render()
                time.sleep(0.1)

            for bbox in env.episode_pred_bboxes:
                test_file.write(f"{','.join(map(str, map(int, bbox)))}\n")

            test_file.close()
            zipf.write(f'./results/res_img_{image_idx}.txt', arcname=f'res_img_{image_idx}.txt')

        zipf.close()

        os.system('python ICDAR13_eval_script/script.py -g=ICDAR13_eval_script/sign_gt.zip -s=results/res.zip -p=\'{\"AREA_RECALL_CONSTRAINT\":0.5}\'')


if __name__ == '__main__':
    ray.init()
    test_env = EnvFactory.create_eval_env("simple")
    register_env("textloc", lambda config: test_env)
    config = {
        "env": "textloc",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "model": {
            "dim": 224,
            "conv_filters": [
                [64, (1, 1), 1],
                [32, (9, 9), 1],
                [32, (8, 8), 4],
                [16, (9, 9), 4],
                [16, (7, 7), 5],
                [8, (2, 2), 2],
            ],
        },
        "framework": "torch",
    }

    agent = DQNTrainer(config=config)
    agent.restore('./checkpoints/simple_cnn_checkpoint')
    evaluate(agent, test_env)

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