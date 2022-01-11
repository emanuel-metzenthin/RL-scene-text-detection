import os
from os import environ

import hydra
import omegaconf
import ray
import torch
from ray import tune
from ray.rllib.agents.dqn import SimpleQTrainer
from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_CONFIG, DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import merge_dicts
from ray.tune import register_env

from DQN import RLLibImageDQN
from NormalizeFilter import NormalizeFilter
from env_factory import EnvFactory
from evaluate import evaluate
from logger import NeptuneLogger


@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg):
    def custom_eval_fn(trainer, eval_workers):
        eval_dataset = cfg.data.dataset if cfg.data.eval_dataset is None else cfg.data.eval_dataset
        if type(cfg.data.eval_path) == omegaconf.listconfig.ListConfig:
            result = {}
            for i, path in enumerate(cfg.data.eval_path):
                gt_file = cfg.data.eval_gt_file[i]
                eval_env = EnvFactory.create_eval_env(eval_dataset, path, cfg.data.json_path, cfg.env.framestacking_mode, ior_mode=cfg.env.ior_mode, playout=cfg.data.eval_full_playout)
                new_result = evaluate(trainer, eval_env, gt_file)
                renamed_result = {f"{gt_file.split('_')[0]}_{k}": v for k, v in new_result.items()}
                result = {**result, **renamed_result}

            return result
        else:
            eval_env = EnvFactory.create_eval_env(eval_dataset, cfg.data.eval_path, cfg.data.json_path, cfg.env.framestacking_mode, cfg.env.ior_mode, cfg.assessor.checkpoint, cfg.data.eval_full_playout)

            return evaluate(trainer, eval_env, cfg.data.eval_gt_file)

    environ['WORKING_DIR'] = os.getcwd()
    ModelCatalog.register_custom_model("image_dqn", RLLibImageDQN)
    register_env("text_localization_env", lambda _: EnvFactory.create_env(cfg.data.dataset, cfg.data.path, cfg.data.json_path, cfg.data.mix_path, cfg, cfg.env.framestacking_mode, cfg.reward.use_cut_area))
    config = {
        "env": "text_localization_env",
        "model": {
            "custom_model": "image_dqn",
            "custom_model_config": {
                "framestacking_mode": cfg.env.framestacking_mode,
                "backbone": cfg.training.backbone
            }
        },
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "buffer_size": cfg.training.replay_buffer_size,
        "train_batch_size": cfg.training.batch_size,
        "optimizer": merge_dicts(
            DQN_CONFIG["optimizer"], {
                "num_replay_buffer_shards": 1,
            }),
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": cfg.training.epsilon.start,
            "final_epsilon": cfg.training.epsilon.end,
            "epsilon_timesteps": cfg.training.epsilon.decay_steps * cfg.training.envs_per_worker,
        },
        "lr": cfg.training.lr,
        "gamma": cfg.training.loss.gamma,
        "num_workers": 0,
        "rollout_fragment_length": 1,
        "learning_starts": 0,
        "framework": "torch",
        "compress_observations": cfg.compress_observations,
        "logger_config": cfg,
        "observation_filter": lambda x: NormalizeFilter(),
        "seed": cfg.training.random_seed,
        "batch_mode": "complete_episodes",
        "log_sys_usage": False,
        "custom_eval_function": custom_eval_fn,
        "evaluation_interval": cfg.training.eval_iterations,
        "render_env": True
    }

    stop = {
        "training_iteration": cfg.training.iterations,
    }

    callbacks = []
    if not cfg.neptune.offline:
        logger = NeptuneLogger(cfg)
        callbacks += (logger,)

    tune.run(SimpleQTrainer, restore=cfg.restore, local_dir=cfg.log_dir, checkpoint_freq=cfg.training.eval_iterations, config=config, stop=stop, callbacks=callbacks)

    ray.shutdown()


if __name__ == "__main__":
    ray.init()

    main()
