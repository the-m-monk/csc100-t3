from enum import IntEnum

# VISION

VISION_WIDTH = 64
VISION_HEIGHT = 64
VISION_COLOUR_CHANNELS = 3


VISION_OUT_LEN = 32

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
    FINISHED = 7


# MAGI

MAGI_TARGETS = (
    DogModelTarget.TUNNEL,
    DogModelTarget.RAMP,
    DogModelTarget.TILE,
)

MAGI_ACTIONS = (
    (
        DogModelAction.FORWARD,
        DogModelAction.BACKWARD,
        DogModelAction.LEFT,
        DogModelAction.RIGHT,
        DogModelAction.TUNNEL,
    ),
    (
        DogModelAction.FORWARD,
        DogModelAction.BACKWARD,
        DogModelAction.LEFT,
        DogModelAction.RIGHT,
        DogModelAction.RAMP,
    ),
    (
        DogModelAction.FORWARD,
        DogModelAction.BACKWARD,
        DogModelAction.LEFT,
        DogModelAction.RIGHT,
        DogModelAction.FINISHED,
    ),
)

MAGI_ACTION_COUNT = len(MAGI_ACTIONS[0])
TARGET_TO_MAGI = {target: magi_idx for magi_idx, target in enumerate(MAGI_TARGETS)}
