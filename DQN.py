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
            nn.Linear(2048, 1024),
            nn.Linear(1024, num_actions)
        )

    def forward(self, X):
        features = self.feature_extractor(X).squeeze()

        return self.dqn(features)