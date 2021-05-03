import json
from copy import deepcopy

import hydra
import neptune.new as neptune
import ray
from omegaconf import DictConfig, OmegaConf

from DQN import ImageDQN
from actor import Actor
from env_factory import EnvFactory
from learner import Learner
from logging_service import LoggingService
from parameter_server import ParameterServer
from replay_buffer import ReplayBuffer
from train import train


@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg: DictConfig):
    logger = None
    if not cfg.neptune.offline:
        logger = LoggingService.remote(cfg)

    env = EnvFactory.create_env(cfg.dataset, cfg)
    dqn = ImageDQN(num_actions=env.action_space.n)
    target_dqn = ImageDQN(num_actions=env.action_space.n)

    replay_buffer = ReplayBuffer.options(name="replay_buffer", lifetime="detached").remote(cfg)
    param_server = ParameterServer.options(name="param_server", lifetime="detached").remote()
    learner = Learner.remote(dqn, target_dqn, replay_buffer, param_server, logger, cfg)
    actors = [Actor.remote(deepcopy(dqn), deepcopy(target_dqn), deepcopy(env), replay_buffer, learner, param_server, logger, cfg, i) for i in range(cfg.apex.num_actors)]

    all_workers = actors + [learner]

    ray.wait([worker.run.remote() for worker in all_workers])


if __name__ == '__main__':
    ray.init()
    main()
