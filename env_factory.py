from typing import Text

from text_localization_environment import TextLocEnv
from torch.utils.data import DataLoader

from assessor import AssessorModel
from dataset.ICDAR_dataset import ICDARDataset
from dataset.assessor_dataset import AssessorDataset
from dataset.sign_dataset import SignDataset
from dataset.simple_dataset import SimpleDataset


class EnvFactory:
    @staticmethod
    def load_dataset(dataset, data_path, split: Text = 'train'):
        if dataset == "icdar2013":
            return ICDARDataset(path=data_path, split=split)
        elif dataset == "sign":
            return SignDataset(path=data_path, split=split)
        elif dataset == "simple":
            return SimpleDataset(path=data_path)
        else:
            raise Exception(f"Dataset name {dataset} not supported.")

    @staticmethod
    def create_env(name, path, cfg, assessor=False):
        dataset = EnvFactory.load_dataset(name, path)
        assessor_model = None

        if assessor:
            assessor_data = AssessorDataset(cfg.assessor_data_path)
            assessor_model = AssessorModel(DataLoader(assessor_data, batch_size=64, shuffle=True))

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=cfg.env.full_playout,
            premasking=cfg.env.premasking,
            max_steps_per_image=cfg.env.steps_per_image,
            bbox_scaling=0,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=cfg.env.termination,
            has_intermediate_reward=cfg.env.intermediate_reward,
            assessor_model=assessor_model
        )

        env.seed(cfg.env.random_seed)

        return env

    @staticmethod
    def create_eval_env(name, path):
        dataset = EnvFactory.load_dataset(name, path, "validation")

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=False,
            premasking=True,
            max_steps_per_image=200,
            bbox_scaling=0,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=False,
            mode='test'
        )

        return env
