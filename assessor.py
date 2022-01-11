import random
from random import randint

import neptune.new as neptune
import optuna
import torch.nn as nn
import torchvision.models as models
from neptune.new.types import File
from torch import sigmoid, softmax
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
    def __init__(self, alpha=True, train_dataloader=None, hidden_1=64, hidden_2=128, hidden_3=256, output=1, dual_image=False):
        super().__init__()
        input_channels = 4 if alpha else 3
        self.resnet = nn.Sequential(
            ResBlock1(input_channels, hidden_1),
            nn.MaxPool2d(2, 2),
            ResBlock1(hidden_1, hidden_2),
            nn.MaxPool2d(2, 2),
            ResBlock1(hidden_2, hidden_3),
            nn.MaxPool2d(2, 2),
            ResBlock3(hidden_3, hidden_3),
            nn.AvgPool2d(3),
            nn.Flatten(),
            nn.Linear(hidden_3, output, bias=False),
            # nn.Sigmoid()
        )
        self.resnet.apply(self.init_weights)
        self.dual_image = dual_image

        # feat_len = hidden_3 * 2 if self.dual_image else hidden_3
        # self.feat = nn.Linear(feat_len, output, bias=False)
        # self.feat.apply(self.init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.train_dataloader = train_dataloader
        if self.train_dataloader:
            self.train_iter = iter(train_dataloader)

    def forward(self, X):
        if self.dual_image:
            repr = [self.resnet(x) for x in X]
            repr = torch.stack(repr).view(-1, 512)
        else:
            repr = self.resnet(X)

        # out = self.feat(repr)

        return repr

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.02)

    def train_one_step(self):
        self.train()

        try:
            input, labels = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            input, labels = next(self.train_iter)

        input = input.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        pred = self(input)

        if len(pred.shape) == 2 and pred.shape[1] == 2:
            # cutting_pred = torch.argmax(softmax(pred[:, 1:], 1), axis=1)
            iou_loss, cut_loss = self.mse(pred[:, 0], labels[:, 0]), self.bce(sigmoid(pred[:, 1]), labels[:, 1])
            loss = iou_loss + cut_loss
        else:
            mse_loss = self.mse(pred, labels)
            loss = mse_loss
        loss.backward()
        self.optimizer.step()

        # print(f"Assessor loss: {mse_loss}")

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
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])

    if optimizer_name == "RAdam":
        optimizer = RAdam(model.parameters(), lr=lr)
    else:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train(train_path, val_path, trial, optimizer, model)


def train(train_path, val_path, trial, optimizer, model, alpha, tightness, dual_image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # model.load_state_dict(torch.load('assessor_model.pt'))

    train_data = AssessorDataset(train_path, alpha, dual_image=dual_image)
    val_data = AssessorDataset(val_path, alpha, split="val", dual_image=dual_image)
    train_loader = DataLoader(train_data, batch_size=64)
    val_loader = DataLoader(val_data, batch_size=64)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    best_loss = None

    for epoch in range(500):

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

                if tightness:
                    pred = model(input)
                    # cutting_pred = torch.argmax(softmax(pred[:, 1:], 1), axis=1)
                    iou_loss, cut_loss = mse(pred[:, 0], labels[:, 0]), mse(pred[:, 1], labels[:, 1])
                    loss = iou_loss + cut_loss
                else:
                    pred = model(input)
                    mse_loss = mse(pred, labels)
                    loss = mse_loss

                loss.backward()
                optimizer.step()

                pred_mins.append(torch.min(pred).detach().cpu())
                pred_maxs.append(torch.max(pred).detach().cpu())
                pred_vars.append(torch.var(pred).detach().cpu())

                train_losses.append(loss.item())
                train_epoch.set_postfix({'loss': np.mean(train_losses)})

            if run:
                run['train/grad_mean'].log(torch.mean(list(model.parameters())[-1].grad))
                run['train/loss'].log(np.mean(train_losses))
                run['train/pred_min'].log(np.min(pred_mins))
                run['train/pred_max'].log(np.max(pred_maxs))
                run['train/pred_var'].log(np.mean(pred_vars))

            del input
            del labels
            torch.cuda.empty_cache()

        with tqdm(val_loader) as val_epoch:
            with torch.no_grad():
                model.eval()
                # log_batch_ids = random.sample(range(len(val_epoch)), 5)
                exp_imgs = []
                exp_ious = []

                for i, (input, labels) in enumerate(val_epoch):
                    input = input.to(device)
                    labels = labels.to(device)
                    pred = model(input).squeeze()

                    # if i in log_batch_ids:
                    #    img_id = random.sample(range(len(pred)), 1)
                    #    exp_imgs.append(ToPILImage()(input[img_id].squeeze()))
                    #    exp_ious.append([pred[img_id].item()])

                    if tightness:
                        cutting_pred = sigmoid(pred[:, 1]) > 0.5
                        # acc = sum(cutting_pred == labels[:, 1]) / len(labels)
                        iou_loss, cut_loss = mse(pred[:, 0], labels[:, 0]), mse(pred[:, 1], labels[:, 1])
                        val_loss = iou_loss + cut_loss
                        # val_epoch.set_postfix({'val_acc': acc})
                    else:
                        mse_loss = mse(pred, labels)
                        val_loss = mse_loss

                    val_losses.append(val_loss.item())
                    mean_val_loss = np.mean(val_losses)

                    val_epoch.set_postfix({'val_loss': mean_val_loss})

                if run:
                    run['val/loss'].log(mean_val_loss)

                if not best_loss or mean_val_loss < best_loss:
                    #plot_example_images(exp_imgs, exp_ious)
                    torch.save(model.state_dict(), 'assessor_model.pt')
                    if run:
                        run['model'].upload('assessor_model.pt')
                    best_loss = mean_val_loss

                if trial:
                    trial.report(best_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

    return best_loss


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
    parser.add_argument("--param_search", action='store_true', required=False)
    parser.add_argument("--no_alpha", action='store_true', required=False)
    parser.add_argument("--tightness", action='store_true', required=False)
    parser.add_argument("--dual_image", action='store_true', required=False)
    parser.add_argument("--neptune_project", type=str, required=False)
    args = parser.parse_args()

    torch.manual_seed(42)

    run = neptune.init(project=args.neptune_project)
    # run = None
    train_path, val_path = args.train_path, args.val_path

    if not args.param_search:
        model = AssessorModel(not args.no_alpha, output=2 if args.tightness else 1, dual_image=args.dual_image)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        train(train_path, val_path, None, optimizer, model, not args.no_alpha, args.tightness, args.dual_image)
    else:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)

        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
