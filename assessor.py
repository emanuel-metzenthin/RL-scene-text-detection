import random
from random import randint

import neptune.new as neptune
import optuna
import torch.nn as nn
import torchvision.models as models
from neptune.new.types import File
from torch import sigmoid
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
from torch import sigmoid
from dataset.assessor_dataset import AssessorDataset
import matplotlib.pyplot as plt
from logger import NeptuneLogger
from radam import RAdam
from numpy import asarray
from torchvision.models import resnet18
# import plotly.express as px

class ResBlock1(nn.Module):
    def __init__(self, ch_in, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=(4, 4), padding=1, stride=2, bias=False)
        self.conv3 = nn.Conv2d(ch_in, ch, kernel_size=(4, 4), padding=1, stride=2, bias=False)
        self.g1 = nn.GroupNorm(32, ch)
        self.g2 = nn.GroupNorm(32, ch)
        self.g3 = nn.GroupNorm(32, ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        h1 = self.g1(self.conv1(x))
        h2 = self.g2(self.conv2(self.relu(h1)))
        h3 = self.g3(self.conv3(residual))
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
        self.g1 = nn.GroupNorm(32, ch)
        self.g2 = nn.GroupNorm(32, ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        h1 = self.g1(self.conv1(self.relu(x)))
        h2 = self.g2(self.conv2(self.relu(h1)))
        h3 = h2 + residual

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
    def __init__(self, train_dataloader=None, hidden_1=64, hidden_2=128, hidden_3=256):
        super().__init__()
        # backbone_model = models.resnet18(pretrained=False)
        # self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
        #
        # self.linear = nn.Linear(backbone_model.fc.in_features, 1)
        self.resnet = nn.Sequential(
            ResBlock1(3, hidden_1),
            nn.MaxPool2d(2, 2),
            ResBlock1(hidden_1, hidden_2),
            nn.MaxPool2d(2, 2),
            ResBlock1(hidden_2, hidden_3),
            nn.MaxPool2d(2, 2),
            ResBlock3(hidden_3, hidden_3),
            nn.AvgPool2d(3),
            nn.Flatten(),
            nn.Linear(hidden_3, 1, bias=False),
            #nn.Sigmoid()
        )
        self.resnet.apply(self.init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader

    def forward(self, X):
        # feat = self.feature_extractor(X)
        # return sigmoid(self.linear(feat.squeeze()))
        out = self.resnet(X)

        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.02)

    def train_one_step(self):
        self.train()

        input, labels = next(iter(self.train_dataloader))
        input = input.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        pred = self(input)
        mse_loss = self.criterion(pred.float(), labels.float())
        mse_loss.backward()
        self.optimizer.step()

        print(f"Assessor loss: {mse_loss}")

    def evaluate_one_epoch(self):
        pass


def define_model(trial):
    model = AssessorModel(hidden_1=trial.suggest_categorical("resblock1_hidden", [32, 64, 128]),
                          hidden_2=trial.suggest_categorical("resblock2_hidden", [64, 128]),
                          hidden_3=trial.suggest_categorical("resblock3_hidden", [128, 256]))
    return model


def objective(trial):
    model = define_model(trial)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RAdam", "SGD"])
    lr = trial.suggest_uniform("lr", 1e-5, 1e-3)

    if optimizer_name == "RAdam":
        optimizer = RAdam(model.parameters(), lr=lr)
    else:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    return train(train_path, val_path, trial, optimizer, model)


def train(train_path, val_path, trial, optimizer, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # model.load_state_dict(torch.load('assessor_model.pt'))

    train_data = AssessorDataset(train_path)
    val_data = AssessorDataset(val_path, split="val")
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    criterion = nn.MSELoss()

    best_loss = None

    for epoch in range(300):

        val_losses = []
        train_losses = []
        pred_mins = []
        pred_maxs = []
        pred_vars = []

        with tqdm(train_loader) as train_epoch:
            model.train()
            for input, labels in train_epoch:
                input = input.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred = model(input)
                mse_loss = criterion(pred, labels)
                loss = mse_loss

                mse_loss.backward()
                optimizer.step()

                pred_mins.append(torch.min(pred).detach().cpu())
                pred_maxs.append(torch.max(pred).detach().cpu())
                pred_vars.append(torch.var(pred).detach().cpu())

                train_losses.append(loss.item())
                train_epoch.set_postfix({'loss': np.mean(train_losses)})

            # if run:
            #     # run['train/iou_distribution'].upload(px.histogram(labels.cpu()))
            #     run['train/grad_mean'].log(torch.mean(list(model.parameters())[-1].grad))
            #     run['train/loss'].log(np.mean(train_losses))
            #     run['train/pred_min'].log(np.min(pred_mins))
            #     run['train/pred_max'].log(np.max(pred_maxs))
            #     run['train/pred_var'].log(np.mean(pred_vars))

            del input
            del labels
            torch.cuda.empty_cache()

        with tqdm(val_loader) as val_epoch:
            with torch.no_grad():
                model.eval()
                log_batch_ids = random.sample(range(len(val_epoch)), 5)
                exp_imgs = []
                exp_ious = []

                for i, (input, labels) in enumerate(val_epoch):
                    input = input.to(device)
                    labels = labels.to(device)
                    pred = model(input)

                    if i in log_batch_ids:
                        img_id = random.sample(range(len(pred)), 1)
                        exp_imgs.append(ToPILImage()(input[img_id].squeeze()))
                        exp_ious.append([pred[img_id].item()])

                    val_loss = criterion(pred, labels)

                    val_losses.append(val_loss.item())
                    mean_val_loss = np.mean(val_losses)

                    val_epoch.set_postfix({'val_loss': mean_val_loss})
                # if run:
                #     run['val/loss'].log(mean_val_loss)

                if not best_loss or mean_val_loss < best_loss:
                    best_loss = mean_val_loss
                    print(best_loss)
                    print(epoch)

                # trial.report(best_loss, epoch)
                # if trial.should_prune():
                #     raise optuna.exceptions.TrialPruned()

    return best_loss

            # if not best_loss or mean_val_loss < best_loss:
            #     plot_example_images(exp_imgs, exp_ious)
            #     torch.save(model.state_dict(), 'assessor_model.pt')
            #     if run:
            #         run['model'].upload('assessor_model.pt')
            #     best_loss = mean_val_loss


def plot_example_images(images, ious):
    fig = plt.figure()

    for i, (img, iou) in enumerate(zip(images, ious), start=1):
        fig.add_subplot(2, 3, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(str(round(iou[0],2)))

    if run:
        run[f'val/example_imgs'].upload(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", default='/home/emanuel/data/iou_samples/train')
    parser.add_argument("val_path", default='/home/emanuel/data/assessor_data2/val')
    args = parser.parse_args()

    # run = neptune.init(project='emanuelm/assessor')
    train_path, val_path = args.train_path, args.val_path
    model = AssessorModel()
    optimizer = RAdam(model.parameters(), lr=1e-4)
    train(train_path, val_path)

    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=100)
    #
    # pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    # complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    #
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))
    #
    # print("Best trial:")
    # trial = study.best_trial
    #
    # print("  Value: ", trial.value)
    #
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
