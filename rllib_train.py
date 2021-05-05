import argparse
import json

from env_factory import EnvFactory
import hydra
import ray
from ray import tune
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from DQN import RLLibImageDQN
from logger import NeptuneLogger


@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg):
    # env = EnvFactory.create_env("sign", cfg)
    ModelCatalog.register_custom_model("imagedqn", RLLibImageDQN)
    register_env("textloc", lambda config: EnvFactory.create_env("sign", cfg))
    config = {

        "env": "textloc",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        # "model": {
        #     "custom_model": "imagedqn",
        # },
        "lr": 1e-4,  # try different lrs
        "num_workers": 6, #cfg.apex.num_actors,  # parallelism
        "framework": "torch",
        "render_env": True,
        "logger_config": {"neptune_project_name": "emanuelm/rl-scene-text-detection"}
    }

    stop = {
        "training_iteration": 1000,
    }

    results = tune.run("APEX", config=config, stop=stop, loggers=tune.logger.DEFAULT_LOGGERS + (NeptuneLogger,),)

    ray.shutdown()


if __name__ == "__main__":
    ray.init()

    main()
