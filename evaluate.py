import os
import time
import zipfile

import numpy as np
import ray
import torch
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.tune.utils import merge_dicts
from text_localization_environment import TextLocEnv
from tqdm import tqdm

from DQN import RLLibImageDQN
from NormalizeFilter import NormalizeFilter
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
                # time.sleep(0.1)

            for bbox in env.episode_pred_bboxes:
                test_file.write(f"{','.join(map(str, map(int, bbox)))}\n")

            test_file.close()
            zipf.write(f'./results/res_img_{image_idx}.txt', arcname=f'res_img_{image_idx}.txt')

        zipf.close()

        os.system('python ICDAR13_eval_script/script.py -g=ICDAR13_eval_script/simple_gt.zip -s=results/res.zip') # -p=\'{\"AREA_RECALL_CONSTRAINT\":0.5}\' ?


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    ray.init()
    test_env = EnvFactory.create_eval_env("simple", args.data_path)
    register_env("textloc", lambda config: test_env)
    ModelCatalog.register_custom_model("imagedqn", RLLibImageDQN)
    config = {
        "env": "textloc",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "model": {
            "custom_model": "imagedqn",
            "custom_model_config": {
                "dueling": True
            }
        },
        "explore": False,
        "observation_filter": lambda x: NormalizeFilter(),
        "framework": "torch",
    }

    agent = DQNTrainer(config=config)
    agent.restore(args.checkpoint_path)
    evaluate(agent, test_env)