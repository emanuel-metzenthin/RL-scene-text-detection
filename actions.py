from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple


def wrap_coord(coord):
    return min(max(coord, 0), 224)


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class Size(Enum):
    SMALLER = 0
    LARGER = 1


class AspectRatio(Enum):
    FLATTER = 0
    TALLER = 1


class Action(ABC):
    def execute(self, alpha, bbox: Tuple):
        x_min, y_min, x_max, y_max = bbox
        alpha_h = alpha * (y_max - y_min)
        alpha_w = alpha * (x_max - x_min)

        return self.change_bbox(alpha_w, alpha_h, bbox)

    @abstractmethod
    def change_bbox(self, alpha_w, alpha_h, bbox):
        pass


class MoveAction(Action):

    def __init__(self, direction: Direction):
        self.direction = direction

    def change_bbox(self, alpha_w, alpha_h, bbox: Tuple):
        x_min, y_min, x_max, y_max = bbox

        if self.direction == Direction.LEFT:
            if (x_min - alpha_w) > 0 and (x_max - alpha_w) > 0:
                x_min -= alpha_w
                x_max -= alpha_w
        elif self.direction == Direction.RIGHT:
            if (x_min + alpha_w) < 224 and (x_max + alpha_w) < 224:
                x_min += alpha_w
                x_max += alpha_w
        elif self.direction == Direction.UP:
            if (y_min - alpha_h) > 0 and (y_max - alpha_h) > 0:
                y_min -= alpha_h
                y_max -= alpha_h
        elif self.direction == Direction.DOWN:
            if (y_min + alpha_h) < 224 and (y_max + alpha_h) < 224:
                x_min += alpha_h
                x_max += alpha_h

        return wrap_coord(x_min), wrap_coord(y_min), wrap_coord(x_max), wrap_coord(y_max)


class ResizeAction(Action):

    def __init__(self, size: Size):
        self.size = size

    def change_bbox(self, alpha_w, alpha_h, bbox: Tuple):
        x_min, y_min, x_max, y_max = bbox

        if self.size == Size.SMALLER:
            x_min += alpha_w
            x_max -= alpha_w
            y_min += alpha_h
            y_max -= alpha_h
        elif self.size == Size.LARGER:
            x_min -= alpha_w
            x_max += alpha_w
            y_min -= alpha_h
            y_max += alpha_h

        return wrap_coord(x_min), wrap_coord(y_min), wrap_coord(x_max), wrap_coord(y_max)


class ChangeAspectAction(Action):

    def __init__(self, aspect_ratio: AspectRatio):
        self.aspect_ratio = aspect_ratio

    def change_bbox(self, alpha_w, alpha_h, bbox: Tuple):
        x_min, y_min, x_max, y_max = bbox

        if self.aspect_ratio == AspectRatio.FLATTER:
            y_min += alpha_h
            y_max -= alpha_h
        elif self.aspect_ratio == AspectRatio.TALLER:
            x_min += alpha_w
            x_max -= alpha_w

        return wrap_coord(x_min), wrap_coord(y_min), wrap_coord(x_max), wrap_coord(y_max)


class TriggerAction(Action):
    def change_bbox(self, alpha_w, alpha_h, bbox):
        return bbox
