from typing import Text

import torch
from torch.utils.data import DataLoader
from text_localization_environment import TextLocEnv
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
    def create_env(name, path, cfg, assessor=False, framestacking_mode=False, use_cut_area=False):
        dataset = EnvFactory.load_dataset(name, path)
        assessor_model = None

        if assessor:
            if cfg.assessor.data_path:
                assessor_data = AssessorDataset(cfg.assessor.data_path)
                assessor_model = AssessorModel(DataLoader(assessor_data, batch_size=64, shuffle=False))
            else:
                assessor_model = AssessorModel()

            if cfg.assessor.checkpoint:
                assessor_model.load_state_dict(torch.load(cfg.assessor.checkpoint, map_location="cpu"))

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=cfg.env.full_playout,
            premasking=cfg.env.premasking,
            max_steps_per_image=cfg.env.steps_per_image,
            bbox_scaling_w=cfg.env.bbox_scaling_width,
            bbox_scaling_h=cfg.env.bbox_scaling_height,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=cfg.env.termination,
            has_intermediate_reward=cfg.reward.intermediate_reward,
            assessor_model=assessor_model,
            train_assessor=assessor_data is not None,
            grayscale=framestacking_mode == 'grayscale',
            use_cut_area=use_cut_area
        )

        env.seed(cfg.env.random_seed)

        return env

    @staticmethod
    def create_eval_env(name, path, framestacking_mode, playout=False):
        dataset = EnvFactory.load_dataset(name, path, "validation")

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=playout,
            premasking=True,
            max_steps_per_image=200,
            bbox_scaling_w=0,
            bbox_scaling_h=0,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=False,
            mode='test',
            grayscale=framestacking_mode == 'grayscale'
        )

        return env
