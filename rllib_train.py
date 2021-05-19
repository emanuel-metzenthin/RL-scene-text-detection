import argparse
import json
from typing import Optional, Type

import torch
from ray.rllib.agents.dqn import SimpleQTFPolicy, ApexTrainer, SimpleQTorchPolicy, SimpleQTrainer
from ray.rllib.utils.typing import TrainerConfigDict

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
from logger import NeptuneLogger


@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg):
    ModelCatalog.register_custom_model("imagedqn", RLLibImageDQN)
    register_env("textloc", lambda config: EnvFactory.create_env(cfg.dataset, cfg.data_path, cfg))
    config = {
        "env": "textloc",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "buffer_size": cfg.env.replay_buffer.size,
        "train_batch_size": cfg.training.batch_size,
        "prioritized_replay": True,
        "model": {
            "dim": 224,
            "conv_filters": [
                [64, (1, 1), 1],
                [32, (9, 9), 1],
                [32, (8, 8), 4],
                [16, (9, 9), 4],
                [16, (7, 7), 5],
                [8, (2, 2), 1],
            ],
            "post_fcnet_hiddens": [1024, 1024, 1024]
        },
        "optimizer": merge_dicts(
            DQN_CONFIG["optimizer"], {
                "num_replay_buffer_shards": 1,
            }),
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": cfg.env.epsilon.start,
            "final_epsilon": cfg.env.epsilon.end,
            "epsilon_timesteps": cfg.env.epsilon.decay_steps * cfg.training.envs_per_worker,
        },
        "lr": 1e-4,  # try different lrs
        "gamma": cfg.training.loss.gamma,
        "num_workers": 1,
        "num_gpus_per_worker": 1 if torch.cuda.is_available() else 0,
        "num_envs_per_worker": cfg.training.envs_per_worker,
        "rollout_fragment_length": 50,
        "learning_starts": 0,
        "framework": "torch",
        # "render_env": True,
        "logger_config": cfg,
    }

    stop = {
        "episode_reward_mean": 70,
    }

    if cfg.custom_model:
        config["model"] = {
            "custom_model": "imagedqn",
            "custom_model_config": {
                "dueling": True
            }
        }
    loggers = tune.logger.DEFAULT_LOGGERS
    if not cfg.neptune.offline:
        loggers += (NeptuneLogger,)

    if cfg.restore:
        tune.run(DQNTrainer, restore=cfg.restore, local_dir=cfg.log_dir, checkpoint_freq=100, config=config, stop=stop, loggers=loggers)
    else:
        tune.run(DQNTrainer, local_dir=cfg.log_dir, checkpoint_freq=100, config=config, stop=stop, loggers=loggers)

    ray.shutdown()


if __name__ == "__main__":
    ray.init()

    main()
