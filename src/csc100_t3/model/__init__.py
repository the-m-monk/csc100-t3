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
        if not 0 <= magi_idx < len(self.magi):
            raise ValueError(f"unknown magi index: {magi_idx}")

        actions = aux.MAGI_ACTIONS[magi_idx]
        try:
            last_action_index = (
                None if last_action is None else actions.index(last_action)
            )
        except ValueError as error:
            raise ValueError(
                f"{last_action.name} is not available to magi {magi_idx}"
            ) from error

        wm = self.magi[magi_idx]

        return wm["action_head"](wm["cnn"](fb), last_action_index)

    def forward_batch(
        self,
        magi_idx: int,
        frames: np.ndarray,
        last_action_indices: Tensor,
    ) -> Tensor:
        if not 0 <= magi_idx < len(self.magi):
            raise ValueError(f"unknown magi index: {magi_idx}")

        wm = self.magi[magi_idx]
        return wm["action_head"].forward_batch(
            wm["cnn"](frames),
            last_action_indices,
        )
