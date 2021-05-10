import json

import neptune
import numpy as np
from ray import tune
from ray.tune.utils import merge_dicts


class NeptuneLogger(tune.logger.Logger):
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

    def _init(self):
        logger_config = self.config.get('logger_config')
        neptune.init('emanuelm/rl-scene-text-detection')
        self.neptune_experiment = neptune.create_experiment(
            name=str(self.trial),  # Gets the trial name.
            params=self.config,
        )

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
                prefixed_key = '/'.join([prefix, key])
                if isinstance(value, float) or isinstance(value, int):
                    self.neptune_experiment.log_metric(
                        prefixed_key, value)
                elif (isinstance(value, np.ndarray) or
                      isinstance(value, np.number)):
                    self.run[prefixed_key].log(float(value))
                # Otherwise ignore