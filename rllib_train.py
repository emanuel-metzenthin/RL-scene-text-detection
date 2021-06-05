import argparse
import json
import os
from os import environ
from typing import Optional, Type

import torch
from ray.rllib.agents.dqn import SimpleQTrainer
from ray.rllib.utils.typing import TrainerConfigDict

from NormalizeFilter import NormalizeFilter
from env_factory import EnvFactory
import hydra
import ray
from ray import tune
from ray.rllib import agents, Policy
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_CONFIG, DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from DQN import RLLibImageDQN
from evaluate import evaluate
from logger import NeptuneLogger

@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg):
    def custom_eval_fn(trainer, eval_workers):
        eval_env = EnvFactory.create_eval_env(cfg.dataset, cfg.eval_data_path)
        return evaluate(trainer, eval_env)

    environ['WORKING_DIR'] = os.getcwd()
    ModelCatalog.register_custom_model("imagedqn", RLLibImageDQN)
    register_env("textloc", lambda config: EnvFactory.create_env(cfg.dataset, cfg.data_path, cfg))
    config = {
        "env": "textloc",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "buffer_size": cfg.env.replay_buffer.size,
        "train_batch_size": cfg.training.batch_size,
        "model": {
            "dim": 224,
            "conv_filters": [
                [64, (1, 1), 1],
                [32, (9, 9), 1],
                [32, (8, 8), 4],
                [16, (9, 9), 4],
                [16, (7, 7), 5],
                [8, (2, 2), 2],
            ]
        },
        "optimizer": merge_dicts(
            DQN_CONFIG["optimizer"], {
                "num_replay_buffer_shards": 1,
            }),
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0, # cfg.env.epsilon.start,
            "final_epsilon": 0, #cfg.env.epsilon.end,
            "epsilon_timesteps": cfg.env.epsilon.decay_steps * cfg.training.envs_per_worker,
        },
        "n_step": 3,
        "num_atoms": 51,
        "v_min": -3,
        "v_max": 70,
        "noisy": True,
        "lr": 1e-4,  # try different lrs
        "gamma": cfg.training.loss.gamma,
        "num_workers": 0,
        "num_gpus_per_worker": 0.5 if torch.cuda.is_available() else 0,
        "num_envs_per_worker": cfg.training.envs_per_worker,
        "rollout_fragment_length": 1,
        "learning_starts": 0,
        "framework": "torch",
        "compress_observations": True,
        "logger_config": cfg,
        "observation_filter": lambda x: NormalizeFilter(),
        "seed": cfg.training.random_seed,
        "batch_mode": "complete_episodes",
        "log_sys_usage": False,
        "custom_eval_function": custom_eval_fn,
        "evaluation_interval": 300
    }

    stop = {
        "episode_reward_mean": 70,
    }

    if cfg.custom_model:
        config["model"] = {
            "custom_model": "imagedqn",
            "custom_model_config": {
                "dueling": False
            }
        }
    callbacks = []
    if not cfg.neptune.offline:
        logger = NeptuneLogger(cfg)
        callbacks += (logger,)

    tune.run(DQNTrainer, restore=cfg.restore, local_dir=cfg.log_dir, checkpoint_freq=300, config=config, stop=stop, callbacks=callbacks)

    ray.shutdown()


if __name__ == "__main__":
    ray.init()

    main()
