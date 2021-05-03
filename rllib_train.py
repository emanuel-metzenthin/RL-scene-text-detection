import argparse
import json

import hydra
import ray
from ray import tune
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from DQN import RLLibImageDQN
from env_factory import EnvFactory
from logger import NeptuneLogger

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="APEX")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)


@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg):
    # env = EnvFactory.create_env("sign", cfg)
    ModelCatalog.register_custom_model("imagedqn", RLLibImageDQN)
    register_env("textloc", lambda config: EnvFactory.create_env("sign", cfg))
    config = {
        "env": "textloc",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 2,
        # "model": {
        #     "custom_model": "imagedqn",
        # },
        "lr": 1e-4,  # try different lrs
        "num_workers": 6, #cfg.apex.num_actors,  # parallelism
        "framework": "torch",
        "logger_config": {"neptune_project_name": "emanuelm/rl-scene-text-detection"}
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop, loggers=tune.logger.DEFAULT_LOGGERS + (NeptuneLogger,),)

    ray.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    main()
