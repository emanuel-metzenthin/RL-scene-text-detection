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
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_CONFIG
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
        "num_gpus": 0.7 if torch.cuda.is_available() else 0,
        "buffer_size": cfg.env.replay_buffer.size,
        "model": {
            "custom_model": "imagedqn",
        },
        # "dueling": False,
        # "double_q": False,
        "optimizer": merge_dicts(
            DQN_CONFIG["optimizer"], {
                "num_replay_buffer_shards": 1,
            }),
        "lr": 1e-4,  # try different lrs
        # "num_workers": cfg.apex.num_actors,  # parallelism
        "num_workers": 1,
        "num_gpus_per_worker": 0.3,
        "num_envs_per_worker": 20,
        "rollout_fragment_length": 30,
        "learning_starts": 30,
        "framework": "torch",
        "logger_config": cfg
    }

    stop = {
        "training_iteration": 1000,
    }

    # def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    #     return SimpleQTorchPolicy
    #
    # CustomTrainer = ApexTrainer.with_updates(
    #     name="SimpleQApex",
    #     default_policy=SimpleQTFPolicy,
    #     get_policy_class=get_policy_class
    # )
    # loggers = tune.logger.DEFAULT_LOGGERS
    # if not cfg.neptune.offline:
    #     loggers += (NeptuneLogger,)
    results = tune.run(SimpleQTrainer, local_dir=cfg.log_dir, checkpoint_freq=100, config=config, stop=stop)

    ray.shutdown()


if __name__ == "__main__":
    ray.init()

    main()
