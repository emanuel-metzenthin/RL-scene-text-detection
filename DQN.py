import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.view_requirement import ViewRequirement
from torch import nn as nn, TensorType
from typing import *
import torch
import torchvision.models as models


class RLLibImageDQN(TorchModelV2, nn.Module):
    def value_function(self) -> TensorType:
        pass

    def __init__(self, obs_space, action_space, num_outputs: int, model_config: dict, name: str, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        custom_model_cfg = model_config["custom_model_config"]
        framestacking_mode = custom_model_cfg["framestacking_mode"] if "framestacking_mode" in custom_model_cfg else False
        backbone = custom_model_cfg["backbone"] if "backbone" in custom_model_cfg else "resnet18"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ImageDQN(backbone=backbone, num_actions=action_space.n, framestacking_mode=framestacking_mode).to(device)

        if framestacking_mode:
            self.view_requirements['obs'] = ViewRequirement(shift="-3:0", space=obs_space)

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        return self.model(input_dict['obs']), []


class ImageDQN(nn.Module):
    BACKBONES = ['resnet18', 'resnet50', 'vgg16']

    def __init__(self, backbone: Text = 'resnet18', num_actions: int = 9, num_history: int = 10, framestacking_mode: bool = None, grayscale: bool = False):
        super().__init__()

        if backbone not in ImageDQN.BACKBONES:
            raise Exception(f'{backbone} not supported.')
        backbone_model = getattr(models, backbone)(pretrained=True)

        if backbone == 'vgg16':
            self.feature_extractor = backbone_model
            self.feature_extractor.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(25088, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096)
            )
            self.feature_extractor_output_size = 4096
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children()))
        else:
            self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
            if backbone == 'resnet50':
                for child in list(self.feature_extractor.children())[:-3]:
                    for param in child.parameters():
                        param.requires_grad = False
            self.feature_extractor_output_size = backbone_model.fc.in_features

        self.framestacking_mode = framestacking_mode
        self.grayscale = grayscale
        if self.framestacking_mode == 'color':
            self.feature_extractor[0] = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif self.framestacking_mode == 'grayscale':
            self.feature_extractor[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.dqn = nn.Sequential(
            nn.Linear(self.feature_extractor_output_size + num_history * num_actions, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)
        )

        self.dqn.apply(self.init_weights)
        self.num_actions = num_actions
        self.num_history = num_history

    def forward(self, X):
        images, histories = X

        if len(images.shape) == 4:
            images = images.permute([0, 3, 1, 2])

            if self.framestacking_mode:
                images = torch.repeat_interleave(images, 4, dim=1)
        else:
            images = images.permute([0, 1, 4, 2, 3])
            histories = torch.reshape(histories[:, -1], (-1, self.num_actions * self.num_history))
            if self.framestacking_mode == 'grayscale':
                images = images.squeeze(2)  # grayscale imgs have 1 channel, resulting in [bs, history, 1, img]
            elif self.framestacking_mode == 'color':
                images = torch.reshape(images, (-1, 12, 224, 224))

        histories = torch.reshape(histories, (-1, self.num_actions * self.num_history))
        features = self.feature_extractor(images).reshape(-1, self.dqn[0].in_features - self.num_actions * self.num_history)

        states = torch.cat((features, histories), dim=1)

        return self.dqn(states)


    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
