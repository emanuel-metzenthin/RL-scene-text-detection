from collections import deque, namedtuple
from typing import Tuple, List
import numpy as np
import ray
import torch

Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


@ray.remote
class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """
    # TODO make replay buffer prioritized
    def __init__(self, cfg) -> None:
        self.buffer = deque(maxlen=cfg.env.replay_buffer.size)
        self.next_batch = None
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.buffer)

    async def append(self, experiences: List[Experience]) -> None:
        """
        Add experience to the buffer
        Args:
            experiences: tuple (state, action, reward, done, new_state)
        """
        self.buffer += experiences
        # print("Replay buffer: received batch")

    def sample(self, batch_size: int) -> Tuple:
        if len(self) < batch_size:
            return
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        states = list(zip(*states))
        next_states = list(zip(*next_states))

        self.next_batch = (states, actions, rewards, dones, next_states)

    async def get_next_batch(self):
        batch = self.next_batch
        self.sample(self.cfg.training.batch_size)

        return batch
