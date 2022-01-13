# Weakly Supervised Scene Text Detection using Deep Reinforcement Learning

This repository contains the setup for all experiments performed in our Paper ...
It is to be used in conjunction with the RL environment [text-localization-environment](https://github.com/emanuel-metzenthin/text-localization-environment), which is linked as a submodule. After cloning do `git submodule init` and `git submodule update` and follow the installation instructions of that repo.

The project is configured using [Hydra](https://hydra.cc/docs/intro/) in the [cfg](/cfg) folder.

## Training

We use [RLLib](https://docs.ray.io/en/master/rllib/index.html) as RL framework. Train the model by executing [rllib_train.py](./rllib_train.py).

Every value in the [cfg](/cfg) folder can be altered by passing it as a CLI argument, while keeping the correct file hierarchy (e.g. `data.path=/data`). The folder _data_ contains templates for different dataset configurations.

Here are explanations for a few example parameters.

| Parameter                    | Description                                 | default         |
|------------------------------|---------------------------------------------|-----------------|
| neptune.offline              | disables logging to neptune.ai              | true            |
| training.iterations          | how long to train                           | 5000            |
| training.epsilon.decay_steps | length of exploration                       | 300000          |
| data.dataset                 | dataset type                                | icdar2013       |
| data.path                    | path to dataset                             | /data/ICDAR2013 |
| data.json_path               | path to json file of data (for SynthText)   | null            |
| data.eval_path               | path to evaluation dataset                  | /data/ICDAR2013 |
| data.eval_gt_file            | gt zip file for IC13/IC15/TIoU eval scripts | icdar13_gt.zip  |

__Training weakly supervised__:

| Parameter           | Description                                                                           |
|---------------------|---------------------------------------------------------------------------------------|
| assessor.data_path  | path to assessor training data for on-the-fly training of the assessor                |
| assessor.checkpoint | path to assessor PyTorch (.pt) file. A pretained model can be downloaded [here](https://bartzi.de/research/weakly-supervised-scene-text-detection). |

__Loading a checkpoint__:

Checkpoints need to be RLLib checkpoint folders. Our best three models (supervised, weakly supervised and semi-supervised) can be downloaded [here](https://bartzi.de/research/weakly-supervised-scene-text-detection).

Set the parameter `restore` to the checkpoint directory. Training will resume from the checkpoint. The training iterations have to be increased, as the checkpoints were made at iteration 15k.


## Testing

Execute [evaluate.py](./evaluate.py).

```
python evaluate.py <checkpoint_dir> <eval_data_dir> <eval_gt_file> --dataset icdar2013 [--framestacking grayscale]
```

## Tips

For IDE debugging change `ray.init()` in [rllib_train.py](./rllib_train.py) to `ray.init(local_mode=True)`.
