import math

from torch import nn
import numpy as np

import csc100_t3.model.aux as aux


class DogVision(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                aux.VISION_COLOUR_CHANNELS,
                aux.VISION_CONV_CHANNELS[0],
                kernel_size=aux.VISION_KERNEL_SIZES[0],
                stride=aux.VISION_STRIDES[0],
                padding=aux.VISION_PADDINGS[0],
            ),
            nn.ReLU(),
            nn.Conv2d(
                aux.VISION_CONV_CHANNELS[0],
                aux.VISION_CONV_CHANNELS[1],
                kernel_size=aux.VISION_KERNEL_SIZES[1],
                stride=aux.VISION_STRIDES[1],
                padding=aux.VISION_PADDINGS[1],
            ),
            nn.ReLU(),
            nn.Conv2d(
                aux.VISION_CONV_CHANNELS[1],
                aux.VISION_CONV_CHANNELS[2],
                kernel_size=aux.VISION_KERNEL_SIZES[2],
                stride=aux.VISION_STRIDES[2],
                padding=aux.VISION_PADDINGS[2],
            ),
            nn.ReLU(),
            nn.Conv2d(
                aux.VISION_CONV_CHANNELS[2],
                aux.VISION_CONV_CHANNELS[3],
                kernel_size=aux.VISION_KERNEL_SIZES[3],
                stride=aux.VISION_STRIDES[3],
                padding=aux.VISION_PADDINGS[3],
            ),
            nn.ReLU(),
        )

        output_height = aux.VISION_HEIGHT
        output_width = aux.VISION_WIDTH

        for kernel, stride, padding in zip(
            aux.VISION_KERNEL_SIZES,
            aux.VISION_STRIDES,
            aux.VISION_PADDINGS,
        ):
            output_height = self.conv_output_size(
                output_height,
                kernel,
                stride,
                padding,
            )

            output_width = self.conv_output_size(
                output_width,
                kernel,
                stride,
                padding,
            )

        flattened_size = aux.VISION_CONV_CHANNELS[-1] * output_height * output_width

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, aux.VISION_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(aux.VISION_HIDDEN_SIZE, aux.VISION_OUT_LEN),
        )

    def conv_output_size(
        input_size,
        kernel_size,
        stride,
        padding,
    ):
        return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

    def forward(self, fb: np.darray):
        fb = torch.from_numpy(fb)

        expected_shape = (
            aux.VISION_HEIGHT,
            aux.VISION_WIDTH,
            aux.VISION_COLOUR_CHANNELS,
        )

        if tuple(fb.shape) != expected_shape:
            raise ValueError(
                f"framebuffer must have shape {expected_shape}, got {tuple(fb.shape)}"
            )

        # hwc to chw
        fb = fb.permute(2, 0, 1)

        # add batch prefix (1)
        fb = fb.unsqueeze(0)

        fb = fb.float() / 255.0

        output = self.head(self.conv(fb))
        return output.squeeze(0)
