import argparse
import torch
from pytorch_lightning import Trainer
from train import RLTraining
from pytorch_lightning.loggers.neptune import NeptuneLogger


parser = argparse.ArgumentParser()
parser.add_argument("--neptune_key", type=str, required=False, help="Neptune.ai API key")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--epochs", type=int, default=50, help="how many epochs to train")
parser.add_argument("--gpus", type=int, default=0, help="number of gpus to train on")
parser.add_argument("--sync_rate", type=int, default=10,
                    help="how many frames do we update the target network")
parser.add_argument("--steps_per_image", type=int, default=200,
                    help="max number of steps per image")
parser.add_argument("--replay_size", type=int, default=1000,
                    help="capacity of the replay buffer")
parser.add_argument("--warm_start_size", type=int, default=1000,
                    help="how many samples do we use to fill our buffer at the start of training")
parser.add_argument("--eps_last_episode", type=int, default=500,
                    help="what episode should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
parser.add_argument("--num_epoch_eval_images", type=int, default=None, help="number of images to evaluate on per epoch")
parser.add_argument("--evaluate_every", type=int, default=10, help="evaluate every n epochs")
parser.add_argument("--max_episode_reward", type=int, default=200,
                    help="max episode reward in the environment")
parser.add_argument("--warm_start_steps", type=int, default=1000,
                    help="max episode reward in the environment")

args = parser.parse_args()

if __name__ == '__main__:'
    rl_training = RLTraining(args)

    trainer = Trainer(gpus=args.gpus, max_epochs=args.epochs)

    if args.neptune_key:
        neptune_logger = NeptuneLogger(
            api_key=args.neptune_key,
            project_name="emanuelm/scene-text-detection",
            params=args.__dict__)
        trainer.logger = neptune_logger

    trainer.fit(rl_training)

