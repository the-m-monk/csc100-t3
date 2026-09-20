import numpy as np
from torch import Tensor, nn

from csc100_t3.model import aux, vision, action_head


class DogModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = vision.DogVision()
        self.action_head = action_head.DogActionHead()

    def forward(
        self,
        fb: np.ndarray,
        target: aux.DogModelTarget,
        last_action: aux.DogModelAction | None,
    ) -> Tensor:
        return self.action_head(self.cnn(fb), target, last_action)
