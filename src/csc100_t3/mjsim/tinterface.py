# training interface

from dataclasses import dataclass
from pathlib import Path
import os
import math
import random

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
        super().__init__()

        self.scene_path = Path(__file__).resolve().parents[3] / "scene" / "main.xml"
        self.scene = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.scene)

        self.cam_renderer = mujoco.Renderer(
            self.scene,
            width=aux.VISION_WIDTH,
            height=aux.VISION_HEIGHT,
        )

        self.dog_pos = DogPos(Vec2(0, 0), 0)
        self.course_state = CourseState(
            Vec2(0, 0),
            0,
            Vec2(0, 0),
            0,
            Vec2(0, 0),
            0,
            Vec2(0, 0),
            0,
        )

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

        self.rseed: int = 0
        self.RESET_COURSE_WATCHDOG_INIT = 100

    def place_relative(
        self,
        prev_x: float,
        prev_y: float,
        prev_yaw: float,
        current_name: str,
        variance: dict,
    ):
        distance = random.uniform(
            variance["dmin"],
            variance["dmax"],
        )

        delta_yaw = random.uniform(
            variance["ymin"],
            variance["ymax"],
        )

        current_yaw = prev_yaw + delta_yaw

        current_x = prev_x + math.cos(current_yaw) * distance
        current_y = prev_y + math.sin(current_yaw) * distance

        current = self.scene.body(current_name)

        current.pos[:2] = [current_x, current_y]

        current.quat[:] = [
            math.cos(current_yaw / 2),
            0,
            0,
            math.sin(current_yaw / 2),
        ]

        return current_x, current_y, current_yaw

    def reset_course(self):
        # set obstacle position

        # dog
        go2_joint = self.scene.joint("go2_joint")
        go2_qadr = self.scene.jnt_qposadr[go2_joint.id]

        self.data.qpos[go2_qadr + 0] = self.DOG_START_COORD.x
        self.data.qpos[go2_qadr + 1] = self.DOG_START_COORD.y
        self.data.qpos[go2_qadr + 3 : go2_qadr + 7] = [
            1,
            0,
            0,
            0,
        ]  # point north/forward

        # tunnel
        tunnel_vec = Vec2(0, 0)
        tunnel_vec.x, tunnel_vec.y, tunnel_yaw = self.place_relative(
            0,
            0,
            0,
            "tunnel",
            self.START_TO_TUNNEL_VARIANCE,
        )

        # ramp
        ramp_vec = Vec2(0, 0)
        ramp_vec.x, ramp_vec.y, ramp_yaw = self.place_relative(
            tunnel_vec.x,
            tunnel_vec.y,
            tunnel_yaw,
            "ramp",
            self.TUNNEL_TO_RAMP_VARIANCE,
        )

        # chest
        chest_vec = Vec2(0, 0)
        chest_vec.x, chest_vec.y, chest_yaw = self.place_relative(
            ramp_vec.x,
            ramp_vec.y,
            ramp_yaw,
            "chest",
            self.RAMP_TO_CHEST_VARIANCE,
        )

        # finish tile
        finish_vec = Vec2(0, 0)
        finish_vec.x, finish_vec.y, finish_yaw = self.place_relative(
            chest_vec.x,
            chest_vec.y,
            chest_yaw,
            "finish_tile",
            self.CHEST_TO_FINISH_VARIANCE,
        )

        self.course_state = CourseState(
            tunnel_vec,
            tunnel_yaw,
            ramp_vec,
            ramp_yaw,
            chest_vec,
            chest_yaw,
            finish_vec,
            finish_yaw,
        )

    def is_keepout_respected(self):
        return False

    def reset(self, rseed: int) -> tuple[StepState, CourseState]:
        self.rseed = rseed
        random.seed(self.rseed)

        mujoco.mj_resetData(self.scene, self.data)

        self.reset_course()

        reset_course_watchdog = self.RESET_COURSE_WATCHDOG_INIT

        while self.is_keepout_respected() == False:
            if reset_course_watchdog == 0:
                # change to actual python error
                print("Error: reset course watchdog")
                exit(1)

            reset_course_watchdog -= 1

            self.reset_course()

        # set course state
        # set dog pos
        # create step state

        """
        mujoco.mj_step(scene, data)

        renderer.update_scene(
            data,
            camera="dog_camera",
        )

        fb = renderer.render()
        """

        # reset course, create random variant, return intial framebuffer and target
        # do intial swing
        ...

    def step(self, cnn: vision.DogVision, action: aux.DogModelAction) -> StepState:
        # do action, return state post action
        # if action is obstacle finish, do obstacle, run spin, and locate next obstacle
        ...
