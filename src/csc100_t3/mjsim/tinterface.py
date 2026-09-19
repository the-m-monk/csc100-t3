# training interface

from dataclasses import dataclass
from pathlib import Path
import os
import math

import numpy as np

os.environ["MUJOCO_GL"] = "egl"
import mujoco

from csc100_t3.model import aux, vision


@dataclass
class StepState:
    dog_pos: DogPos
    fb: np.ndarray
    target: aux.DogModelTarget
    remaining_steps: int
    # done == (max steps reached or reached finish tile), sucess == (reached finish tile and !(max steps reached))
    done: bool
    sucess: bool


@dataclass
class Vec2:
    x: float
    y: float


@dataclass
class CourseState:
    tunnel_coord: Vec2
    tunnel_yaw: float

    ramp_coord: Vec2
    ramp_yaw: float

    block_coord: Vec2
    block_yaw: float

    finish_tile_coord: Vec2
    finish_tile_yaw: float


@dataclass
class DogPos:
    coord: Vec2
    yaw: float


class TrainingSimulator:
    def __init__(self):
        super.__init__()

        self.scene_path = Path(__file__).resolve().parents[3] / "scene" / "main.xml"
        self.scene = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.scene)

        self.cam_renderer = mujoco.Renderer(
            self.scene,
            width=aux.VISION_WIDTH,
            height=aux.VISION_HEIGHT,
        )

        self.dog_pos = DogPos(0, 0, 0)
        self.course_state = CourseState(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # reset config
        self.DOG_START_COORD = Vec2(0, 0)

        self.KEEPOUT_RADIAL_SPACING = 0.5
        self.DOG_KEEPOUT = (0.7 / 2) + self.KEEPOUT_RADIAL_SPACING
        self.TUNNEL_KEEPOUT = (1.3 / 2) + self.KEEPOUT_RADIAL_SPACING
        self.RAMP_KEEPOUT = (1.2 / 2) + self.KEEPOUT_RADIAL_SPACING
        self.CHEST_KEEPOUT = (0.7 / 2) + self.KEEPOUT_RADIAL_SPACING
        self.FINISH_KEEPOUT = (0.3 / 2) + self.KEEPOUT_RADIAL_SPACING

        self.START_TO_TUNNEL_VARIANCE = {
            "dmin": self.DOG_KEEPOUT + self.TUNNEL_KEEPOUT,
            "dmax": 10,
            "ymin": math.radians(-90),
            "ymax": math.radians(90),
        }

        self.TUNNEL_TO_RAMP_VARIANCE = {
            "dmin": self.TUNNEL_KEEPOUT + self.RAMP_KEEPOUT,
            "dmax": 10,
            "ymin": math.radians(-90),
            "ymax": math.radians(90),
        }

        self.RAMP_TO_CHEST_VARIANCE = {
            "dmin": self.RAMP_KEEPOUT + self.CHEST_KEEPOUT,
            "dmax": 10,
            "ymin": math.radians(-90),
            "ymax": math.radians(90),
        }

        self.CHEST_TO_FINISH_VARIANCE = {
            "dmin": self.CHEST_KEEPOUT + self.FINISH_KEEPOUT,
            "dmax": 10,
            "ymin": math.radians(-90),
            "ymax": math.radians(90),
        }

        # movement

        # forward/backward variance
        # left/right variance

    def reset(self, rseed: int) -> tuple[StepState, CourseState]:
        """
        mujoco.mj_step(scene, data)

        renderer.update_scene(
            data,
            camera="dog_camera",
        )

        fb = renderer.render()
        """

        # set obstacle position
        # check keepout violation
        # set course state
        # set dog pos
        # create step state

        # reset course, create random variant, return intial framebuffer and target
        # do intial swing
        ...

    def step(self, cnn: vision.DogVision, action: aux.DogModelAction) -> StepState:
        # do action, return state post action
        # if action is obstacle finish, do obstacle, run spin, and locate next obstacle
        ...
