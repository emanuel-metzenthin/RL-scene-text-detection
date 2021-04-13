import torch
from torch import nn as nn
from typing import *
import torchvision.models as models


class ImageDQN(nn.Module):
    BACKBONES = ['resnet18', 'resnet50']

    def __init__(self, backbone: Text = 'resnet50', num_actions: int = 9, num_history: int = 10):
        super().__init__()

        if backbone not in ImageDQN.BACKBONES:
            raise Exception(f'{backbone} not supported.')

        backbone_model = getattr(models, backbone)(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.dqn = nn.Sequential(
            nn.Linear(2048 + num_history * num_actions, 1024),
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

        if images.shape[1] != 3:
            images = images.permute([0, 3, 1, 2])

        histories = torch.reshape(histories, (-1, self.num_actions * self.num_history))

        features = self.feature_extractor(images).reshape(-1, 2048)
        states = torch.cat((features, histories), dim=1)

        return self.dqn(states)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
