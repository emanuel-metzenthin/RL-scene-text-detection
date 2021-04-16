import hydra
import neptune
from omegaconf import DictConfig
from train import train


@hydra.main(config_path="cfg", config_name="config.yml")
def main(cfg: DictConfig):
    if not cfg.neptune.offline:
        neptune.init(api_token=cfg.neptune.key, project_qualified_name='emanuelm/scene-text-detection')
        neptune.create_experiment(cfg.neptune.run_name, params=cfg)

    train(cfg)


if __name__ == '__main__':
    main()
