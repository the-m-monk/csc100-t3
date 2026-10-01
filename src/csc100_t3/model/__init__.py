import numpy as np
from torch import Tensor, nn

from csc100_t3.model import vision, action_head
from csc100_t3.model import common as aux


class DogModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.magi = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cnn": vision.DogVision(),
                        "action_head": action_head.DogActionHead(),
                    }
                )
                for _ in aux.MAGI_TARGETS
            ]
        )

    def forward(
        self,
        magi_idx: int,
        fb: np.ndarray,
        last_action: aux.DogModelAction | None,
    ) -> Tensor:
        wm = self.magi[magi_idx]

        return wm["action_head"](wm["cnn"](fb), last_action)
