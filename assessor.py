import neptune.new as neptune
import torch.nn as nn
import torchvision.models as models
from torch import sigmoid
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch import sigmoid
from dataset.assessor_dataset import AssessorDataset
from radam import RAdam


class ResBlock1(nn.Module):
    def __init__(self, ch_in, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=(4, 4), padding=1, stride=2, bias=False)
        self.conv3 = nn.Conv2d(ch_in, ch, kernel_size=(4, 4), padding=1, stride=2, bias=False)
        self.group_norm = nn.GroupNorm(32, ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        h1 = self.group_norm(self.conv1(x))
        h2 = self.group_norm(self.conv2(self.relu(h1)))
        h3 = self.group_norm(self.conv3(residual))
        h4 = h2 + h3

        return h4


class ResBlock2(nn.Module):
    def __init__(self, ch_in, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=(4, 4), padding=1, stride=2, bias=False)
        self.conv3 = nn.Conv2d(ch_in, ch, kernel_size=(4, 4), padding=1, stride=2, bias=False)
        self.group_norm = nn.GroupNorm(32, ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        h1 = self.group_norm(self.conv1(self.relu(x)))
        h2 = self.group_norm(self.conv2(self.relu(h1)))
        h3 = self.group_norm(self.conv3(residual))
        h4 = h2 + h3

        return h4


class ResBlock3(nn.Module):
    def __init__(self, ch_in, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=(3, 3), padding=1, bias=False)
        self.conv3 = nn.Conv2d(ch_in, ch, kernel_size=(3, 3), padding=1, bias=False)
        self.group_norm = nn.GroupNorm(32, ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        h1 = self.group_norm(self.conv1(self.relu(x)))
        h2 = self.group_norm(self.conv2(self.relu(h1)))
        h3 = h2 + residual  # how not to use this conv? but still add h2 and residual
        # h4 = h2 + h3

        return h3


class AddCoord(nn.Module):
    def forward(self, X):
        b, c, w, h = X.shape
        x_coord = np.repeat(np.arange(0, w), h)
        x_coord = torch.from_numpy(x_coord.reshape((1, w, h))).unsqueeze(0)
        x_coord = x_coord.repeat(b, 1, 1, 1)

        y_coord = np.array([np.repeat(ih, w) for ih in range(h)])
        y_coord = torch.from_numpy(y_coord.reshape((1, w, h))).unsqueeze(0)
        y_coord = y_coord.repeat(b, 1, 1, 1)

        return torch.cat([X, x_coord, y_coord], 1)


class AssessorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone_model = models.resnet18(pretrained=False)
        # self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
        #
        # self.linear = nn.Linear(backbone_model.fc.in_features, 1)

        self.add_coord = AddCoord()
        self.resnet = nn.Sequential(
            ResBlock1(3, 64),
            nn.MaxPool2d(2),
            ResBlock1(64, 128),
            nn.MaxPool2d(2),
            ResBlock1(128, 256),
            nn.MaxPool2d(2),
            ResBlock3(256, 256),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(4096, 1),
            #nn.Sigmoid()
        )
        self.resnet.apply(self.init_weights)

    def forward(self, X):
        # feat = self.feature_extractor(X)
        # return sigmoid(self.linear(feat.squeeze()))
        out = self.resnet(X)

        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal(m.weight, 0.02)


def train():
    EPS = 0.0001
    model = AssessorModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.load_state_dict(torch.load('assessor_model.pt'))

    train_data = AssessorDataset('/home/emanuel/data/iou_samples/train')
    val_data = AssessorDataset('/home/emanuel/data/assessor_data2/val', split="val")
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0.1)

    best_loss = None

    for epoch in range(300):
        val_losses = []
        train_losses = []
        pred_mins = []
        pred_maxs = []
        pred_vars = []

        model.train()
        with tqdm(train_loader) as train_epoch:
            for input, labels in train_epoch:
                input = input.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred = model(input)
                mse_loss = criterion(pred.float(), labels.float())
                loss = mse_loss

                mse_loss.backward()
                optimizer.step()

                pred_mins.append(torch.min(pred).detach().cpu())
                pred_maxs.append(torch.max(pred).detach().cpu())
                pred_vars.append(torch.var(pred).detach().cpu())

                train_losses.append(loss.item())
                train_epoch.set_postfix({'loss': np.mean(train_losses)})

            run['train/loss'].log(np.mean(train_losses))
            run['train/pred_min'].log(np.min(pred_mins))
            run['train/pred_max'].log(np.max(pred_maxs))
            run['train/pred_var'].log(np.mean(pred_vars))

        with tqdm(val_loader) as val_epoch:
            model.eval()
            for input, labels in val_epoch:
                input = input.to(device)
                labels = labels.to(device)
                pred = model(input)
                val_loss = criterion(pred, labels)

                val_losses.append(val_loss.item())
                mean_val_loss = np.mean(val_losses)

                val_epoch.set_postfix({'val_loss': mean_val_loss})

            run['val/loss'].log(mean_val_loss)

        if not best_loss or mean_val_loss < best_loss:
            torch.save(model.state_dict(), 'assessor_model.pt')
            run['model'].upload('assessor_model.pt')
            best_loss = mean_val_loss


if __name__ == '__main__':
    run = neptune.init(run="AS-208", project='emanuelm/assessor')
    train()

