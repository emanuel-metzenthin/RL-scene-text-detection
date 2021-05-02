import json

import neptune.new as neptune
import ray


@ray.remote
class LoggingService:
    def __init__(self, cfg):
        self.current_parameters = None

        self.run = neptune.init(run=cfg.neptune.run_id, api_token=cfg.neptune.key, project='emanuelm/rl-scene-text-detection', name=cfg.neptune.run_name)
        neptune_dict = json.loads(str(cfg).replace("\'", '"').replace('True', "true").replace("False", "false").replace("None", "null"))
        self.run['parameters'] = neptune_dict

    def log(self, name, value):
        self.run[name].log(value)
