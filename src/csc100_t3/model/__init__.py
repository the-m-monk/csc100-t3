import numpy as np
from torch import nn

from csc100_t3.model import aux, vision, action_head


class DogModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = vision.DogVision()
        self.action_head = action_head.DogActionHead()

    def forward(self, fb: np.ndarray, target: aux.DogModelTarget) -> aux.DogModelAction:
        return self.action_head.forward(self.cnn.forward(fb), target)
