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
from PIL import Image, ImageDraw
from ray.rllib.agents.dqn import SimpleQTrainer
import uuid
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


def evaluate(agent, env, gt_file='simple_gt.zip'):
    num_images = len(env.image_paths)
    cwd = os.environ['WORKING_DIR']
    id = str(uuid.uuid4())[:8]
    dir_name_13 = f'{cwd}/results_ic13_{id}'
    dir_name_15 = f'{cwd}/results_ic15_{id}'

    with tqdm(range(num_images)) as timages:
        _DUMMY_AGENT_ID = "agent0"

        os.makedirs(dir_name_13)
        os.makedirs(dir_name_15)
        os.makedirs("./examples/trajectories", exist_ok=True)

        zipf_ic13 = zipfile.ZipFile(f'{dir_name_13}/res.zip', 'w', zipfile.ZIP_DEFLATED)
        zipf_ic15 = zipfile.ZipFile(f'{dir_name_15}/res.zip', 'w', zipfile.ZIP_DEFLATED)

        avg_ious = []
        for image_idx in timages:
            test_file_ic13 = open(f'{dir_name_13}/res_img_{image_idx}.txt', 'w+')
            test_file_ic15 = open(f'{dir_name_15}/res_img_{image_idx}.txt', 'w+')

            obs = {_DUMMY_AGENT_ID: env.reset(image_index=image_idx)}
            done = False
            episode_image = env.episode_image.copy()
            image_draw = ImageDraw.Draw(episode_image)
            step_count = 0

            while not done:
                action = agent.compute_action(obs[_DUMMY_AGENT_ID], explore=False)
                # do step in the environment
                obs[_DUMMY_AGENT_ID], r, done, _ = env.step(action)
                env.render()
                if image_idx % 20 == 0:
                    step_count += 1
                    if not os.path.isdir(f"./examples/trajectories/{image_idx}"):
                        os.makedirs(f"./examples/trajectories/{image_idx}")
                    Image.fromarray(env.render(mode='rgb_array')).save(f"./examples/trajectories/{image_idx}/{step_count}.png")

            for bbox in env.episode_pred_bboxes:
                image_draw.rectangle(bbox.tolist(), outline=(255, 0, 0), width=3)

                bbox = list(map(int, bbox))
                if bbox[0] < 0 and bbox[2] < 0 or bbox[1] < 0 and bbox[3] < 0:
                    continue
                test_file_ic13.write(f"{','.join(map(str, bbox))}\n")  # ICDAR 2013
                test_file_ic15.write(f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[0]},{bbox[3]}\n')  # ICDAR 2015

            if image_idx % 20 == 0:
                episode_image.save(f"./examples/{image_idx}.png")
                episode_image.save(f"./examples/trajectories/{image_idx}_final.png")

            if env.episode_trigger_ious:
                avg_ious.append(np.mean(env.episode_trigger_ious))

            test_file_ic13.close()
            test_file_ic15.close()

            zipf_ic13.write(f'{dir_name_13}/res_img_{image_idx}.txt', arcname=f'res_img_{image_idx}.txt')
            zipf_ic15.write(f'{dir_name_15}/res_img_{image_idx}.txt', arcname=f'res_img_{image_idx}.txt')

        zipf_ic13.close()
        zipf_ic15.close()

        stdout_ic13 = subprocess.run(['python', f'{cwd}/ICDAR13_eval_script/script.py',
                                      f'-g={cwd}/ICDAR13_eval_script/{gt_file}', f'-s={dir_name_13}/res.zip'],
                                     stdout=subprocess.PIPE).stdout
        results_ic13 = re.search('\{(.*)\}', str(stdout_ic13)).group(0)
        results_ic13 = json.loads(results_ic13)
        ic13_prec = results_ic13['precision']
        ic13_rec = results_ic13['recall']
        ic13_f1 = results_ic13['hmean']

        stdout_ic15 = subprocess.run(['python', f'{cwd}/ICDAR15_eval_script/script.py',
                                      f'-g={cwd}/ICDAR15_eval_script/{gt_file}', f'-s={dir_name_15}/res.zip'],
                                     stdout=subprocess.PIPE).stdout
        results_ic15 = re.search('\{(.*)\}', str(stdout_ic15)).group(0)
        results_ic15 = json.loads(results_ic15)
        ic15_prec = results_ic15['precision']
        ic15_rec = results_ic15['recall']
        ic15_f1 = results_ic15['hmean']

        stdout_tiou = subprocess.run(['python', f'{cwd}/TIoU_eval_script/script.py',
                                      f'-g={cwd}/ICDAR15_eval_script/{gt_file}', f'-s={dir_name_15}/res.zip'],
                                     stdout=subprocess.PIPE).stdout
        tiou_prec = re.search('tiouPrecision: (\d\.?\d{0,3})', str(stdout_tiou)).group(1)
        tiou_rec = re.search('tiouRecall: (\d\.?\d{0,3})', str(stdout_tiou)).group(1)
        tiou_f1 = re.search('tiouHmean: (\d\.?\d{0,3})', str(stdout_tiou)).group(1)

        results = {
            'ic13_precision': ic13_prec,
            'ic13_recall': ic13_rec,
            'ic13_f1_score': ic13_f1,
            'ic15_precision': ic15_prec,
            'ic15_recall': ic15_rec,
            'ic15_f1_score': ic15_f1,
            'tiou_precision': float(tiou_prec),
            'tiou_recall': float(tiou_rec),
            'tiou_f1_score': float(tiou_f1),
            'avg_iou': np.mean(avg_ious)
        }

        print(f"IC13 results:\nprecision: {ic13_prec}, recall: {ic13_rec}, f1: {ic13_f1}")
        print(f"IC15 results:\nprecision: {ic15_prec}, recall: {ic15_rec}, f1: {ic15_f1}")
        print(f"TIoU results:\nprecision: {tiou_prec}, recall: {tiou_rec}, f1: {tiou_f1}")
        print(f"Average IoU: {np.mean(avg_ious)}")

        shutil.rmtree(dir_name_13)
        shutil.rmtree(dir_name_15)



        return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("gt_file", type=str)
    parser.add_argument("--dataset", type=str, default="simple")
    parser.add_argument("--framestacking", type=str, default=None)
    parser.add_argument("--playout", type=bool, default=False)
    args = parser.parse_args()

    ray.init()
    test_env = EnvFactory.create_eval_env(args.dataset, args.data_path, None, framestacking_mode=args.framestacking, playout=args.playout)
    register_env("textloc", lambda config: test_env)
    ModelCatalog.register_custom_model("imagedqn", RLLibImageDQN)
    config = {
        "env": "textloc",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "model": {
            "custom_model": "imagedqn",
            "custom_model_config": {
                "dueling": False,
                "framestacking_mode": args.framestacking
            }
        },
        "num_gpus_per_worker": 1,
        "explore": False,
        "observation_filter": lambda x: NormalizeFilter(),
        "framework": "torch",
    }

    agent = SimpleQTrainer(config=config)
    agent.restore(args.checkpoint_path)
    evaluate(agent, test_env, args.gt_file)
