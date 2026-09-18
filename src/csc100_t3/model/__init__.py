from enum import IntEnum

import numpy as np
from torch import nn

import csc100_t3.model.vision as vision


class DogModelAction(IntEnum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    TUNNEL = 5
    RAMP = 6
    BLOCK_FINISH = 7  # get model to call this after navigating the block itself (to trigger next-obj scan)
    FINISHED = 8


class DogModelTarget(IntEnum):
    TUNNEL = 0
    RAMP = 1
    BLOCK = 2
    TILE = 3


class DogModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = vision.DogVision()

    def forward(self, fb: np.ndarray, target: DogModelTarget) -> DogModelAction:

        # CNN (128x128x3) -> vec (correlated with DogModelCnnOutIdx)
        # MLP (cnn vec, target) -> DogModelAction

        return DogModelAction.FINISHED
