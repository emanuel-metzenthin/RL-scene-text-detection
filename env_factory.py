from typing import Text

import torch
from torch.utils.data import DataLoader
from text_localization_environment import TextLocEnv
from assessor import AssessorModel
from dataset.ICDAR_dataset import ICDARDataset
from dataset.LSVT_dataset import LSVTDataset
from dataset.assessor_dataset import AssessorDataset
from dataset.coco_text_dataset import COCOTextDataset
from dataset.icdar_mix_dataset import LSVTIcdarMixDataset
from dataset.sign_dataset import SignDataset
from dataset.sign_icdar_mix_dataset import SignIcdarMixDataset
from dataset.simple_dataset import SimpleDataset


class EnvFactory:
    @staticmethod
    def load_dataset(dataset, data_path, json_path, mix_path=None, mix_labels=True, split: Text = 'train'):
        if dataset == "icdar2013":
            return ICDARDataset(path=data_path, json_path=None, split=split)
        elif dataset == "sign":
            return SignDataset(path=data_path, json_path=json_path, split=split)
        elif dataset == "simple":
            return SimpleDataset(path=data_path, json_path=None)
        elif dataset == "sign_icdar_mix":
            return SignIcdarMixDataset(path=data_path, json_path=json_path, mix_path=mix_path, mix_labels=mix_labels)
        elif dataset == "lsvt_icdar_mix":
            return LSVTIcdarMixDataset(path=data_path, json_path=json_path, mix_path=mix_path, mix_labels=mix_labels)
        elif dataset == "lsvt":
            return LSVTDataset(path=data_path, json_path=None)
        elif dataset == "coco":
            return COCOTextDataset(path=data_path, json_path=json_path, split=split)
        else:
            raise Exception(f"Dataset name {dataset} not supported.")

    @staticmethod
    def create_env(name, path, json_path, mix_path, cfg, framestacking_mode=False, use_cut_area=False):
        dataset = EnvFactory.load_dataset(name, path, json_path, mix_path)
        assessor_data = None
        assessor_model = None

        if cfg.assessor.data_path:
            assessor_data = AssessorDataset(cfg.assessor.data_path, alpha=True, dual_image=cfg.assessor.dual_image)
            assessor_model = AssessorModel(train_dataloader=DataLoader(assessor_data, batch_size=64, shuffle=False),
                                           alpha=True, dual_image=cfg.assessor.dual_image, output=cfg.assessor.output)

        if cfg.assessor.checkpoint:
            if assessor_model is None:
                assessor_model = AssessorModel(alpha=True, dual_image=cfg.assessor.dual_image, output=cfg.assessor.output)
            assessor_model.load_state_dict(torch.load(cfg.assessor.checkpoint, map_location="cpu"))

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=cfg.env.full_playout,
            premasking=cfg.env.premasking,
            max_steps_per_image=cfg.env.steps_per_image,
            bbox_scaling_w=cfg.env.bbox_scaling_width,
            bbox_scaling_h=cfg.env.bbox_scaling_height,
            bbox_transformer='base',
            ior_marker_type=cfg.env.ior_mode,
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
    def create_eval_env(name, path, json_path, framestacking_mode, ior_mode='cross', assessor_checkpoint=None, playout=False):
        dataset = EnvFactory.load_dataset(name, path, json_path, split="validation")
        assessor_model = None

        if assessor_checkpoint:
            assessor_model = AssessorModel(alpha=True)
            assessor_model.load_state_dict(torch.load(assessor_checkpoint, map_location="cpu"))

        env = TextLocEnv(
            dataset.images, dataset.gt,
            playout_episode=playout,
            premasking=True,
            max_steps_per_image=200,
            bbox_scaling_w=0,
            bbox_scaling_h=0,
            bbox_transformer='base',
            ior_marker_type=ior_mode,
            has_termination_action=False,
            assessor_model=assessor_model,
            mode='test',
            grayscale=framestacking_mode == 'grayscale'
        )

        return env
