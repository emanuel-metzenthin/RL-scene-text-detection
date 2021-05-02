from typing import Text

from text_localization_environment import TextLocEnv

from dataset.ICDAR_dataset import ICDARDataset
from dataset.sign_dataset import SignDataset
from dataset.simple_dataset import SimpleDataset


class EnvFactory:
    @staticmethod
    def load_dataset(dataset, split: Text = 'validation'):
        if dataset == "icdar2013":
            return ICDARDataset(path='../data/ICDAR2013', split=split)
        elif dataset == "sign":
            return SignDataset(path='../data/600_3_signs_3_words', split=split)
        elif dataset == "simple":
            return SimpleDataset(path='../data/dataset-generator')
        else:
            raise Exception(f"Dataset name {dataset} not supported.")

    @staticmethod
    def create_env(name, cfg):
        dataset = EnvFactory.load_dataset(name)

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=cfg.env.full_playout,
            premasking=cfg.env.premasking,
            max_steps_per_image=cfg.env.steps_per_image,
            bbox_scaling=0,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=cfg.env.termination,
            has_intermediate_reward=cfg.env.intermediate_reward
        )

        return env
