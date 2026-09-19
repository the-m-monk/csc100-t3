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

    chest_coord: Vec2
    chest_yaw: float

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
        self.MAX_STEPS = 500
        self.step_state = StepState(self.DOG_START_COORD, np.array([]), aux.DogModelTarget.TUNNEL, 0, False)

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
        objs = [
            (self.DOG_START_COORD, self.DOG_KEEPOUT),
            (self.course_state.tunnel_coord, self.TUNNEL_KEEPOUT),
            (self.course_state.ramp_coord, self.RAMP_KEEPOUT),
            (self.course_state.chest_coord, self.CHEST_KEEPOUT),
            (self.course_state.finish_tile_coord, self.FINISH_KEEPOUT),
        ]

        for i, a in enumerate(objs):
            for b in objs[i + 1:]:

                dx = a[0].x - b[0].x
                dy = a[0].y - b[0].y

                dsq = dx**2 + dy**2
                mksq = (a[1] + b[1])**2

                if mksq > dsq:
                    return False

        return True

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

        mujoco.mj_step(self.scene, self.data)

        self.cam_renderer.update_scene(
            self.data,
            camera="go2_camera",
        )

        self.step_state = StepState(
            self.dog_pos,
            self.cam_renderer.render(),
            aux.DogModelTarget.TUNNEL,
            self.MAX_STEPS,
            False,
        )

        return (self.step_state, self.course_state)

    def step(self, action: aux.DogModelAction) -> StepState:
        if self.step_state.remaining_steps > 0:
            self.step_state.remaining_steps -= 1

        if self.step_state.done:
            return self.step_state

        match action:
            case aux.DogModelAction.FORWARD:
                # move forward with variance
                ...
            case aux.DogModelAction.BACKWARD:
                # move backward with variance
                ...
            case aux.DogModelAction.LEFT:
                # move left with variance
                ...
            case aux.DogModelAction.RIGHT:
                # move right with variance
                ...
            case aux.DogModelAction.TUNNEL:
                # go2 pos = tunnel centre + keepout + 0.1, go2 yaw = tunnel yaw
                ...
            case aux.DogModelAction.RAMP:
                # go2 pos = ramp centre + keepout + 0.1, go2 yaw = ramp yaw
                ...
            case aux.DogModelAction.FINISHED:
                self.step_state.remaining_steps = 0
                self.step_state.done = True
                return self.step_state

        

