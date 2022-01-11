import json
from typing import Dict

import neptune.new as neptune
import numpy as np
from neptune.new.internal.operation import AssignBool
from ray import tune
from ray.tune.utils import merge_dicts


class NeptuneLogger(tune.logger.LoggerCallback):
    """RLlib Neptune logger.
    Example usage:
    ```
    import ray
    from ray import tune
    ray.init()
    tune.run(
        "PPO",
        stop={"episode_reward_mean": 200},
        config={
            "env": "CartPole-v0",
            "num_gpus": 0,
            "num_workers": 1,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "logger_config": {"neptune_project_name": '<user name>/sandbox'},
        },
        loggers=tune.logger.DEFAULT_LOGGERS + (NeptuneLogger,),
    )
    ```
    """

    def __init__(self, cfg):
        self.run = neptune.init(run=cfg.neptune.run_id, project=cfg.neptune.project, name=cfg.neptune.run_name)
        neptune_dict = json.loads(str(cfg).replace("\'", '"').replace('True', "true").replace("False", "false").replace("None", "null"))
        self.run['parameters'] = neptune_dict

    @staticmethod
    def dict_multiple_get(dict_, indices):
        """Access the nested value."""
        if not isinstance(indices, list):
            indices = [indices]

        value = dict_
        index = None
        for index in indices:
            try:
                value = value[index]
            except KeyError:
                print("Skipping", indices)
                return {}

        if isinstance(value, dict):
            return value
        else:
            return {index: value}

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        # if trial in self._trial_files:
        list_to_traverse = [
            [],
            ['custom_metrics'],
            ['evaluation'],
            ['info', 'num_steps_trained'],
            ['info', 'learner'],
            ['info', 'exploration_infos', 0],
            ['info', 'exploration_infos', 1],
            ['info', 'learner', "default_policy"]
        ]

        for indices in list_to_traverse:
            res_ = self.dict_multiple_get(result, indices)
            prefix = '/'.join([str(idx) for idx in indices])
            for key, value in res_.items():
                if isinstance(value, AssignBool):
                    continue
                prefixed_key = '/'.join([prefix, key])
                if isinstance(value, float) or isinstance(value, int):
                    self.run[prefixed_key].log(value)
                elif (isinstance(value, np.ndarray) or
                      isinstance(value, np.number)):
                    self.run[prefixed_key].log(float(value))
                # Otherwise ignore

    def log_trial_save(self, trial):
        checkpoint_path = trial.checkpoint.value
        self.run['checkpoint_path'] = checkpoint_path

    def on_result(self, result):
        list_to_traverse = [
            [],
            ['custom_metrics'],
            ['evaluation'],
            ['info', 'num_steps_trained'],
            ['info', 'learner'],
            ['info', 'exploration_infos', 0],
            ['info', 'exploration_infos', 1],
            ['info', 'learner', "default_policy"]
        ]

        for indices in list_to_traverse:
            res_ = self.dict_multiple_get(result, indices)
            prefix = '/'.join([str(idx) for idx in indices])
            for key, value in res_.items():
                if isinstance(value, AssignBool):
                    continue
                prefixed_key = '/'.join([prefix, key])
                if isinstance(value, float) or isinstance(value, int):
                    self.run[prefixed_key].log(value)
                elif (isinstance(value, np.ndarray) or
                      isinstance(value, np.number)):
                    self.run[prefixed_key].log(float(value))
                # Otherwise ignore

    def upload(self, name, artefact):
        self.run[name].upload(artefact)

