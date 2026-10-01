import torch
from torch import nn, Tensor

import csc100_t3.model.common as aux


class DogActionHead(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = (
            aux.VISION_OUT_LEN + len(aux.DogModelAction) + 1
        )  # +1 for "no previous action"

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
                len(aux.DogModelAction),
            ),
        )

    def forward(
        self,
        vision: Tensor,
        last_action: aux.DogModelAction | None,
    ) -> Tensor:
        last_action_index = (
            0
            if last_action is None
            else list(aux.DogModelAction).index(last_action) + 1
        )
        last_action_input = nn.functional.one_hot(
            torch.tensor(last_action_index, device=vision.device),
            num_classes=len(aux.DogModelAction) + 1,
        ).float()

        if vision.shape != (aux.VISION_OUT_LEN,):
            raise ValueError(
                f"vision must have shape ({aux.VISION_OUT_LEN},), "
                f"got {tuple(vision.shape)}"
            )

        x = torch.cat((vision, last_action_input))
        return self.mlp(x)
