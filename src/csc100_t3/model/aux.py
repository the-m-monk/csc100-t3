from enum import IntEnum

# VISION

VISION_WIDTH = 128
VISION_HEIGHT = 128
VISION_COLOUR_CHANNELS = 3


class DogModelCnnOutIdx(IntEnum):
    TUNNEL_VIS = 0  # 0..1
    TUNNEL_BEAR = 1  # -1..1
    TUNNEL_DIS = 2  # 0..1
    TUNNEL_ALIGN = 3  # -1..1

    RAMP_VIS = 4  # 0..1
    RAMP_BEAR = 5  # -1..1
    RAMP_DIS = 6  # 0..1
    RAMP_ALIGN = 7  # -1..1

    BLOCK_VIS = 8  # 0..1
    BLOCK_BEAR = 9  # -1..1
    BLOCK_DIS = 10  # 0..1

    TILE_VIZ = 11  # 0..1
    TILE_BEAR = 12  # -1..1
    TILE_DIS = 13  # 0..1


VISION_OUT_LEN = len(DogModelCnnOutIdx)

VISION_CONV_CHANNELS = (16, 24, 32, 48)
VISION_KERNEL_SIZES = (5, 3, 3, 3)
VISION_STRIDES = (2, 2, 2, 2)
VISION_PADDINGS = (2, 1, 1, 1)
VISION_HIDDEN_SIZE = 128

# ACTOR HEAD


class DogModelTarget(IntEnum):
    TUNNEL = 0
    RAMP = 1
    BLOCK = 2
    TILE = 3


ACTOR_HIDDEN_SIZE = 64


class DogModelAction(IntEnum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    TUNNEL = 5
    RAMP = 6
    BLOCK_FINISH = 7  # get model to call this after navigating the block itself (to trigger next-obj scan)
    FINISHED = 8
