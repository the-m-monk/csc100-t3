from enum import Enum

import numpy as np


class DogModelAction(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    TUNNEL = 5
    RAMP = 6
    BLOCK_FINISH = 7 #get model to call this after navigating the block itself (to trigger next-obj scan)
    FINISHED = 8


FB_DIMENSIONS = (480, 640, 3)


class DogModel:
    def forward(self, fb: np.ndarray) -> DogModelAction:
        if isinstance(np.ndarray, fb):
            raise TypeError("fb must be a np.ndarray")

        if fb.shape != FB_DIMENSIONS:
            raise ValueError(f"fb.shape != {FB_DIMENSIONS}")

        if fb.dtype != np.uint8:
            raise TypeError("fb.dtype != np.uint8")

        return DogModelAction.FINISHED
