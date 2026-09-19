import torch
from torch import nn, Tensor

import csc100_t3.model.aux as aux


class DogActionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                aux.VISION_OUT_LEN + len(aux.DogModelTarget),
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

    def forward(self, vision: Tensor, target: aux.DogModelTarget):
        target = nn.functional.one_hot(
            torch.tensor(target.value),
            num_classes=len(aux.DogModelTarget),
        ).float()

        if vision.shape != (aux.VISION_OUT_LEN,):
            raise ValueError(
                f"vision must have shape ({aux.VISION_OUT_LEN},), "
                f"got {tuple(vision.shape)}"
            )

        x = torch.cat((vision, target))
        return self.mlp(x)
