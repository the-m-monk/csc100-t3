from enum import Enum, IntEnum

import numpy as np

class DogModelAction(Enum):
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

class DogModel:
    def forward(self, fb: np.ndarray, target: DogModelTarget) -> DogModelAction:

        return DogModelAction.FINISHED
