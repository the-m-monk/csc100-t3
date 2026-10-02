import torch
from torch import nn, Tensor

import csc100_t3.model.common as aux


class DogActionHead(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = (
            aux.VISION_OUT_LEN
            + aux.MAGI_ACTION_COUNT
            + 1  # "no previous action"
            + aux.RELATIVE_HEADING_BIN_COUNT
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                input_size,
                aux.ACTOR_HIDDEN_SIZE,
            ),
            nn.ReLU(),
            nn.Linear(
                aux.ACTOR_HIDDEN_SIZE,
                aux.ACTOR_HIDDEN_SIZE,
            ),
            nn.ReLU(),
            nn.Linear(
                aux.ACTOR_HIDDEN_SIZE,
                aux.MAGI_ACTION_COUNT,
            ),
        )

    def forward(
        self,
        vision: Tensor,
        last_action_index: int | None,
        relative_heading_bin: int,
    ) -> Tensor:
        one_hot_index = 0 if last_action_index is None else last_action_index + 1
        last_action_input = nn.functional.one_hot(
            torch.tensor(one_hot_index, device=vision.device),
            num_classes=aux.MAGI_ACTION_COUNT + 1,
        ).float()
        if not 0 <= relative_heading_bin < aux.RELATIVE_HEADING_BIN_COUNT:
            raise ValueError(f"unknown relative heading bin: {relative_heading_bin}")
        heading_input = nn.functional.one_hot(
            torch.tensor(relative_heading_bin, device=vision.device),
            num_classes=aux.RELATIVE_HEADING_BIN_COUNT,
        ).float()

        if vision.shape != (aux.VISION_OUT_LEN,):
            raise ValueError(
                f"vision must have shape ({aux.VISION_OUT_LEN},), "
                f"got {tuple(vision.shape)}"
            )

        x = torch.cat((vision, last_action_input, heading_input))
        return self.mlp(x)

    def forward_batch(
        self,
        vision: Tensor,
        last_action_indices: Tensor,
        relative_heading_bins: Tensor,
    ) -> Tensor:
        if vision.ndim != 2 or vision.shape[1] != aux.VISION_OUT_LEN:
            raise ValueError(
                f"vision must have shape (N, {aux.VISION_OUT_LEN}), "
                f"got {tuple(vision.shape)}"
            )

        last_action_input = nn.functional.one_hot(
            last_action_indices,
            num_classes=aux.MAGI_ACTION_COUNT + 1,
        ).float()
        heading_input = nn.functional.one_hot(
            relative_heading_bins,
            num_classes=aux.RELATIVE_HEADING_BIN_COUNT,
        ).float()
        return self.mlp(torch.cat((vision, last_action_input, heading_input), dim=1))
