from typing import Tuple

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from actions import MoveAction, Direction, Size, ResizeAction, ChangeAspectAction, AspectRatio, TriggerAction, Action
from utils import get_closest_gt


class Environment:
    ALPHA = 0.2
    NU = 0.2
    P = 0.2

    ACTION_SET = [
        MoveAction(Direction.LEFT),
        MoveAction(Direction.UP),
        MoveAction(Direction.RIGHT),
        MoveAction(Direction.DOWN),
        ResizeAction(Size.SMALLER),
        ResizeAction(Size.LARGER),
        ChangeAspectAction(AspectRatio.FLATTER),
        ChangeAspectAction(AspectRatio.TALLER),
        TriggerAction()
    ]

    def __init__(self, dataset):
        self.dataset = dataset
        self.iter_data = None
        self.current_image = None
        self.current_gt = None
        self.current_bbox = (0, 0, 224, 224)
        self.current_steps = 0
        self.current_reward = 0
        self.current_state = None
        self.reset()

    def reset(self):
        self.iter_data = iter(DataLoader(self.dataset, batch_size=1, collate_fn=self.dataset.collate_fn))
        self.next_episode()

    def next_episode(self):
        try:
            self.current_image, self.current_gt = next(self.iter_data)
        except StopIteration:
            self.reset()
        self.current_image, self.current_gt = self.current_image[0], self.current_gt[0]
        self.current_bbox = (0, 0, 224, 224)
        self.current_steps = 0
        self.current_reward = 0
        self.current_state = self.current_image.clone()

    def step(self, action: Action) -> Tuple[Tensor, float, bool, Tensor]:
        old_state = self.current_state.clone()
        done = False
        reward = 0
        self.current_steps += 1

        if isinstance(action, TriggerAction) or self.current_steps > 20:
            closest_gt, iou = get_closest_gt(self.current_bbox, self.current_gt)
            reward = self.calculate_trigger_reward(iou, self.current_steps)
            self.current_reward += reward
            self.next_episode()
            done = True
        else:
            self.current_bbox = action.execute(self.ALPHA, self.current_bbox)
            x1, y1, x2, y2 = self.current_bbox
            self.current_state = self.current_image[:, int(y1):int(y2), int(x1):int(x2)]
            self.current_state = Resize((224, 224))(self.current_state)

        return old_state, reward, done, self.current_state

    def calculate_trigger_reward(self, iou, steps):
        return self.NU * iou - steps * self.P
