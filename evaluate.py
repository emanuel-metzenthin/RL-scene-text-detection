import os
import time
import zipfile

import numpy as np
import ray
import torch
from ray.rllib.agents.dqn import ApexTrainer
from ray.tune import register_env
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
                obs[_DUMMY_AGENT_ID], _, done, _ = env.step(action[_DUMMY_AGENT_ID])
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
    test_env = EnvFactory.create_eval_env("sign")
    register_env("textloc", lambda config: test_env)
    config = {
        "env": "textloc",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        # "buffer_size": cfg.env.replay_buffer.size,
        # "model": {
        #     "custom_model": "imagedqn",
        # },
        # "dueling": False,
        # "double_q": False,
        # "optimizer": merge_dicts(
        #     DQN_CONFIG["optimizer"], {
        #         "num_replay_buffer_shards": 1,
        #     }),
        # "lr": 1e-4,  # try different lrs
        "num_workers": 1,
        "framework": "torch",
        # "render_env": True,
        # "logger_config": cfg
    }

    agent = ApexTrainer(config=config)
    agent.restore('./checkpoints/checkpoint-1000')
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