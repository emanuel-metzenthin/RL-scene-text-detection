import json

import hydra
import neptune.new as neptune
from omegaconf import DictConfig, OmegaConf
from train import train


@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg: DictConfig):
    run = None
    if not cfg.neptune.offline:
        run = neptune.init(api_token=cfg.neptune.key, project='emanuelm/rl-scene-text-detection', name=cfg.neptune.run_name)
        neptune_dict = json.loads(str(cfg).replace("\'", '"').replace('True', "true").replace("False", "false").replace("None", "null"))
        run['parameters'] = neptune_dict
    train(cfg, run)


if __name__ == '__main__':
    main()
