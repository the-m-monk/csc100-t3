import numpy as np
from torch import Tensor, nn

from csc100_t3.model import vision, action_head
from csc100_t3.model import common as aux


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

    def forward_batch(
        self,
        frames: np.ndarray,
        targets: Tensor,
        last_action_indices: Tensor,
    ) -> Tensor:
        return self.action_head.forward_batch(
            self.cnn(frames), targets, last_action_indices
        )
