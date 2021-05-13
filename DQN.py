import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn as nn, TensorType
from typing import *
import torch
import torchvision.models as models


class RLLibImageDQN(TorchModelV2, nn.Module):
    def value_function(self) -> TensorType:
        pass

    def __init__(self, obs_space, action_space, num_outputs: int, model_config: dict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        dueling = model_config["dueling"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ImageDQN(dueling=dueling, num_actions=action_space.n).to(device)

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        return self.model(input_dict['obs']), []


class ImageDQN(nn.Module):
    BACKBONES = ['resnet18', 'resnet50']

    def __init__(self, backbone: Text = 'resnet50', dueling=False, num_actions: int = 9, num_history: int = 10):
        super().__init__()

        if backbone not in ImageDQN.BACKBONES:
            raise Exception(f'{backbone} not supported.')

        backbone_model = getattr(models, backbone)(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.dqn = nn.Sequential(
            nn.Linear(backbone_model.fc.in_features + num_history * num_actions, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 if dueling else num_actions)
        )

        self.dqn.apply(self.init_weights)
        self.num_actions = num_actions
        self.num_history = num_history

    def forward(self, X):
        images, histories = X
        if images.shape[1] != 3:
            images = images.permute([0, 3, 1, 2])
        histories = torch.reshape(histories, (-1, self.num_actions * self.num_history))

        features = self.feature_extractor(images).reshape(-1, self.dqn[0].in_features - self.num_actions * self.num_history)
        states = torch.cat((features, histories), dim=1)

        return self.dqn(states)



    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
