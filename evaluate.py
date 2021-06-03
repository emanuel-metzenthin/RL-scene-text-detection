import os
import shutil
import time
import zipfile
import subprocess
import numpy as np
import ray
import json
import re
import torch
from ray.rllib.agents.dqn import SimpleQTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.tune.utils import merge_dicts
from text_localization_environment import TextLocEnv
from tqdm import tqdm

from DQN import RLLibImageDQN
from NormalizeFilter import NormalizeFilter
from env_factory import EnvFactory
from logger import NeptuneLogger


def evaluate(agent, env):
    num_images = len(env.image_paths)
    cwd = os.environ['WORKING_DIR']

    with tqdm(range(num_images)) as timages:
        _DUMMY_AGENT_ID = "agent0"

        if os.path.exists(f'{cwd}/results_ic13'):
            shutil.rmtree(f'{cwd}/results_ic13/')
        if os.path.exists(f'{cwd}/results_ic15'):
            shutil.rmtree(f'{cwd}/results_ic15/')

        os.makedirs(f'{cwd}/results_ic13')
        os.makedirs(f'{cwd}/results_ic15')

        zipf_ic13 = zipfile.ZipFile(f'{cwd}/results_ic13/res.zip', 'w', zipfile.ZIP_DEFLATED)
        zipf_ic15 = zipfile.ZipFile(f'{cwd}/results_ic15/res.zip', 'w', zipfile.ZIP_DEFLATED)

        avg_ious = []
        for image_idx in timages:
            test_file_ic13 = open(f'{cwd}/results_ic13/res_img_{image_idx}.txt', 'w+')
            test_file_ic15 = open(f'{cwd}/results_ic15/res_img_{image_idx}.txt', 'w+')

            obs = {_DUMMY_AGENT_ID: env.reset(image_index=image_idx)}
            done = False

            while not done:
                action = agent.compute_action(obs[_DUMMY_AGENT_ID])
                # do step in the environment
                obs[_DUMMY_AGENT_ID], r, done, _ = env.step(action)
                # env.render()

            for bbox in env.episode_pred_bboxes:
                bbox = list(map(int, bbox))
                if bbox[0] < 0 and bbox[2] < 0 or bbox[1] < 0 and bbox[3] < 0:
                    continue
                test_file_ic13.write(f"{','.join(map(str, bbox))}\n")  # ICDAR 2013
                test_file_ic15.write(f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[0]},{bbox[3]}')  # ICDAR 2015

            if env.episode_trigger_ious:
                avg_ious.append(np.mean(env.episode_trigger_ious))

            test_file_ic13.close()
            test_file_ic15.close()

            zipf_ic13.write(f'{cwd}/results_ic13/res_img_{image_idx}.txt', arcname=f'res_img_{image_idx}.txt')
            zipf_ic15.write(f'{cwd}/results_ic15/res_img_{image_idx}.txt', arcname=f'res_img_{image_idx}.txt')

        zipf_ic13.close()
        zipf_ic15.close()

        stdout_ic13 = subprocess.run(['python', f'{cwd}/ICDAR13_eval_script/script.py',
                                      f'-g={cwd}/ICDAR13_eval_script/simple_gt.zip', f'-s={cwd}/results_ic13/res.zip'],
                                     stdout=subprocess.PIPE).stdout
        results_ic13 = re.search('\{(.*)\}', str(stdout_ic13)).group(0)
        results_ic13 = json.loads(results_ic13)
        ic13_prec = results_ic13['precision']
        ic13_rec = results_ic13['recall']
        ic13_f1 = results_ic13['hmean']

        stdout_ic15 = subprocess.run(['python', f'{cwd}/ICDAR15_eval_script/script.py',
                                      f'-g={cwd}/ICDAR15_eval_script/simple_gt.zip', f'-s={cwd}/results_ic15/res.zip'],
                                     stdout=subprocess.PIPE).stdout
        results_ic15 = re.search('\{(.*)\}', str(stdout_ic15)).group(0)
        results_ic15 = json.loads(results_ic15)
        ic15_prec = results_ic15['precision']
        ic15_rec = results_ic15['recall']
        ic15_f1 = results_ic15['hmean']

        results = {
            'ic13_precision': ic13_prec,
            'ic13_recall': ic13_rec,
            'ic13_f1_score': ic13_f1,
            'ic15_precision': ic15_prec,
            'ic15_recall': ic15_rec,
            'ic15_f1_score': ic15_f1,
            'avg_iou': np.mean(avg_ious)
        }

        print(f"IC13 results:\nprecision: {ic13_prec}, recall: {ic13_rec}, f1: {ic13_f1}")
        print(f"IC15 results:\nprecision: {ic15_prec}, recall: {ic15_rec}, f1: {ic15_f1}")
        print(f"Average IoU: {np.mean(avg_ious)}")

        return results


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
                "dueling": False
            }
        },
        "num_gpus_per_worker": 1,
        "explore": False,
        "observation_filter": lambda x: NormalizeFilter(),
        "framework": "torch",
    }

    agent = SimpleQTrainer(config=config)
    agent.restore(args.checkpoint_path)
    evaluate(agent, test_env)
