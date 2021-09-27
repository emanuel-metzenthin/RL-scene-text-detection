from typing import Text

import torch
from torch.utils.data import DataLoader
from text_localization_environment import TextLocEnv
from assessor import AssessorModel
from dataset.ICDAR_dataset import ICDARDataset
from dataset.assessor_dataset import AssessorDataset
from dataset.sign_dataset import SignDataset
from dataset.sign_icdar_mix_dataset import SignIcdarMixDataset
from dataset.simple_dataset import SimpleDataset


class EnvFactory:
    @staticmethod
    def load_dataset(dataset, data_path, json_path, mix_path=None, split: Text = 'train'):
        if dataset == "icdar2013":
            return ICDARDataset(path=data_path, json_path=None, split=split)
        elif dataset == "sign":
            return SignDataset(path=data_path, json_path=json_path, split=split)
        elif dataset == "simple":
            return SimpleDataset(path=data_path, json_path=None)
        elif dataset == "sign_icdar_mix":
            return SignIcdarMixDataset(path=data_path, json_path=json_path, mix_path=mix_path)
        else:
            raise Exception(f"Dataset name {dataset} not supported.")

    @staticmethod
    def create_env(name, path, json_path, mix_path, cfg, framestacking_mode=False, use_cut_area=False):
        dataset = EnvFactory.load_dataset(name, path, json_path, mix_path)
        assessor_data = None
        assessor_model = None

        if cfg.assessor.data_path:
            assessor_data = AssessorDataset(cfg.assessor.data_path, alpha=True)
            assessor_model = AssessorModel(train_dataloader=DataLoader(assessor_data, batch_size=64, shuffle=False), alpha=True)

        if cfg.assessor.checkpoint:
            if assessor_model is None:
                assessor_model = AssessorModel(alpha=True)
            assessor_model.load_state_dict(torch.load(cfg.assessor.checkpoint, map_location="cpu"))

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=cfg.env.full_playout,
            premasking=cfg.env.premasking,
            premasking_decay=cfg.env.premasking_decay,
            explore_force_trigger=cfg.training.explore_force_trigger,
            max_steps_per_image=cfg.env.steps_per_image,
            bbox_scaling_w=cfg.env.bbox_scaling_width,
            bbox_scaling_h=cfg.env.bbox_scaling_height,
            bbox_transformer='base',
            ior_marker_type='cross',
            has_termination_action=cfg.env.termination,
            has_intermediate_reward=cfg.reward.intermediate_reward,
            has_repeat_penalty=cfg.reward.repeat_penalty,
            assessor_model=assessor_model,
            train_assessor=assessor_data is not None,
            grayscale=framestacking_mode == 'grayscale',
            use_cut_area=use_cut_area
        )

        env.seed(cfg.env.random_seed)

        return env

    @staticmethod
    def create_eval_env(name, path, json_path, framestacking_mode, playout=False):
        dataset = EnvFactory.load_dataset(name, path, json_path, "validation")

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
