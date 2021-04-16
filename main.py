import argparse
import neptune
from train import train

parser = argparse.ArgumentParser()
parser.add_argument("--neptune_key", type=str, required=False, help="Neptune.ai API key")
parser.add_argument("--run_name", type=str, default='experiment', help="Neptune.ai run name")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--epochs", type=int, default=50, help="how many epochs to train")
parser.add_argument("--gpus", type=int, default=0, help="number of gpus to train on")
parser.add_argument("--sync_rate", type=int, default=10,
                    help="how many frames do we update the target network")
parser.add_argument("--steps_per_image", type=int, default=200,
                    help="max number of steps per image")
parser.add_argument("--warm_start_size", type=int, default=1000,
                    help="how many samples do we use to fill our buffer at the start of every epoch")
parser.add_argument("--steps_per_epoch", type=int, default=1000,
                    help="how many steps to perform per training epoch")
parser.add_argument("--replay_size", type=int, default=1000,
                    help="capacity of the replay buffer")
parser.add_argument("--eps_last_episode", type=int, default=500,
                    help="what episode should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
parser.add_argument("--replay_buffer_sample_size", type=int, default=200, help="how many sample to take from replay buffer")
parser.add_argument("--num_epoch_eval_images", type=int, default=None, help="number of images to evaluate on per epoch")
parser.add_argument("--evaluate_every", type=int, default=10, help="evaluate every n epochs")
parser.add_argument("--max_episode_reward", type=int, default=200,
                    help="max episode reward in the environment")
parser.add_argument("--update_every", type=int, default=5,
                    help="after how many agent steps to update DQN")
parser.add_argument("--checkpoint", type=str, required=False,
                    help="resume from checkpoint")
parser.add_argument("--backbone", type=str, required=False,
                    help="resnet18 or resnet50")
parser.add_argument("--full_playout", action='store_true',
                    help="play episode until max_steps")

args = parser.parse_args()

if __name__ == '__main__':
    if args.neptune_key:
        neptune.init(api_token=args.neptune_key, project_qualified_name='emanuelm/scene-text-detection')
        neptune.create_experiment(args.run_name, params=args.__dict__)

    train(args)

