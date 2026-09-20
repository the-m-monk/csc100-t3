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
    done: bool
    last_action: aux.DogModelAction | None


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

        self.MIN_SPACING = 2

        self.START_TO_TUNNEL_VARIANCE = {
            "dmin": max(self.DOG_KEEPOUT + self.TUNNEL_KEEPOUT, self.MIN_SPACING),
            "dmax": 8,
            "ymin": math.radians(-70),
            "ymax": math.radians(70),
        }

        self.TUNNEL_TO_RAMP_VARIANCE = {
            "dmin": max(self.TUNNEL_KEEPOUT + self.RAMP_KEEPOUT, self.MIN_SPACING),
            "dmax": 8,
            "ymin": math.radians(-70),
            "ymax": math.radians(70),
        }

        self.RAMP_TO_CHEST_VARIANCE = {
            "dmin": max(self.RAMP_KEEPOUT + self.CHEST_KEEPOUT, self.MIN_SPACING),
            "dmax": 8,
            "ymin": math.radians(-70),
            "ymax": math.radians(70),
        }

        self.CHEST_TO_FINISH_VARIANCE = {
            "dmin": max(self.CHEST_KEEPOUT + self.FINISH_KEEPOUT, self.MIN_SPACING),
            "dmax": 8,
            "ymin": math.radians(-70),
            "ymax": math.radians(70),
        }

        # movement
        # https://github.com/Yaocheng-yan/Unitree-go2-Navi/blob/main/技术文档.md
        self.MOV_FB = 0.25  # not framebuffer, forward and back
        self.MOV_R = math.radians(15)

        self.rseed: int = 0
        self.RESET_COURSE_WATCHDOG_INIT = 100
        self.MAX_STEPS = 300
        self.step_state = StepState(
            DogPos(self.DOG_START_COORD, 0),
            np.array([]),
            aux.DogModelTarget.TUNNEL,
            0,
            False,
            None,
        )

        self.POST_OBSTACLE_GAP = 0.3
        dog_rear_x = self.mesh_x_bounds("go2_body_mesh")[0]
        self.TUNNEL_EXIT_OFFSET = (
            self.mesh_x_bounds("tunnel_body")[1] - dog_rear_x + self.POST_OBSTACLE_GAP
        )
        self.RAMP_EXIT_OFFSET = (
            self.mesh_x_bounds("ramp_body")[1] - dog_rear_x + self.POST_OBSTACLE_GAP
        )

    def mesh_x_bounds(self, geom_name: str) -> tuple[float, float]:
        geom = self.scene.geom(geom_name)
        mesh_id = geom.dataid[0]
        first = self.scene.mesh_vertadr[mesh_id]
        count = self.scene.mesh_vertnum[mesh_id]
        vertices = self.scene.mesh_vert[first : first + count]

        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, geom.quat)
        x = vertices @ rotation.reshape(3, 3)[0] + geom.pos[0]
        return float(x.min()), float(x.max())

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

    def is_keepout_respected(
        self, dog_pos_vec2=None, course=None, disable_radial_spacing=False
    ):
        if dog_pos_vec2 == None:
            dog_pos_vec2 = self.DOG_START_COORD

        if course == None:
            course = self.course_state

        drs = 0

        if disable_radial_spacing:
            drs = self.KEEPOUT_RADIAL_SPACING

        objs = [
            (dog_pos_vec2, self.DOG_KEEPOUT - drs),
            (course.tunnel_coord, self.TUNNEL_KEEPOUT - drs),
            (course.ramp_coord, self.RAMP_KEEPOUT - drs),
            (course.chest_coord, self.CHEST_KEEPOUT - drs),
            (course.finish_tile_coord, self.FINISH_KEEPOUT - drs),
        ]

        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                dx = a[0].x - b[0].x
                dy = a[0].y - b[0].y

                dsq = dx**2 + dy**2
                mksq = (a[1] + b[1]) ** 2

                if mksq > dsq:
                    return False

        return True

    # relative
    def move_go2(self, d: Vec2, y: float):
        go2_joint = self.scene.joint("go2_joint")
        go2_qadr = self.scene.jnt_qposadr[go2_joint.id]

        w, x, qy, z = self.data.qpos[go2_qadr + 3 : go2_qadr + 7]

        yaw = math.atan2(
            2.0 * (w * z + x * qy),
            1.0 - 2.0 * (qy * qy + z * z),
        )

        dx = d.x * math.cos(yaw) - d.y * math.sin(yaw)
        dy = d.x * math.sin(yaw) + d.y * math.cos(yaw)

        self.data.qpos[go2_qadr + 0] += dx
        self.data.qpos[go2_qadr + 1] += dy

        yaw += y

        self.data.qpos[go2_qadr + 3 : go2_qadr + 7] = [
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2),
        ]

        self.step_state.dog_pos.coord.x += dx
        self.step_state.dog_pos.coord.y += dy
        self.step_state.dog_pos.yaw = yaw

    # absolute
    def clear_go2_velocity(self):
        go2_joint = self.scene.joint("go2_joint")
        go2_dadr = self.scene.jnt_dofadr[go2_joint.id]
        self.data.qvel[go2_dadr : go2_dadr + 6] = 0

    def set_go2_pos(self, d: Vec2, y: float):
        go2_joint = self.scene.joint("go2_joint")
        go2_qadr = self.scene.jnt_qposadr[go2_joint.id]

        self.data.qpos[go2_qadr + 0] = d.x
        self.data.qpos[go2_qadr + 1] = d.y

        self.data.qpos[go2_qadr + 3 : go2_qadr + 7] = [
            math.cos(y / 2),
            0,
            0,
            math.sin(y / 2),
        ]
        self.clear_go2_velocity()

        self.step_state.dog_pos.coord.x = d.x
        self.step_state.dog_pos.coord.y = d.y
        self.step_state.dog_pos.yaw = y

    def reset(self, rseed: int) -> tuple[StepState, CourseState]:
        self.rseed = rseed
        random.seed(self.rseed)
        np.random.seed(self.rseed)

        mujoco.mj_resetData(self.scene, self.data)

        self.dog_pos = DogPos(
            Vec2(self.DOG_START_COORD.x, self.DOG_START_COORD.y),
            0,
        )

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
            None,
        )

        return (self.step_state, self.course_state)

    def step(self, action: aux.DogModelAction) -> StepState:
        if self.step_state.remaining_steps > 0:
            self.step_state.remaining_steps -= 1
        else:
            self.step_state.done = True

        if self.step_state.done:
            return self.step_state

        match action:
            case aux.DogModelAction.FORWARD:
                self.move_go2(Vec2(self.MOV_FB, 0), 0)
            case aux.DogModelAction.BACKWARD:
                self.move_go2(Vec2(-1 * self.MOV_FB, 0), 0)
            case aux.DogModelAction.LEFT:
                self.move_go2(Vec2(0, 0), -1 * self.MOV_R)
            case aux.DogModelAction.RIGHT:
                self.move_go2(Vec2(0, 0), self.MOV_R)
            case aux.DogModelAction.TUNNEL:
                self.set_go2_pos(
                    Vec2(
                        self.course_state.tunnel_coord.x
                        + math.cos(self.course_state.tunnel_yaw)
                        * self.TUNNEL_EXIT_OFFSET,
                        self.course_state.tunnel_coord.y
                        + math.sin(self.course_state.tunnel_yaw)
                        * self.TUNNEL_EXIT_OFFSET,
                    ),
                    self.course_state.tunnel_yaw,
                )
                self.step_state.target = aux.DogModelTarget.RAMP
            case aux.DogModelAction.RAMP:
                self.set_go2_pos(
                    Vec2(
                        self.course_state.ramp_coord.x
                        + math.cos(self.course_state.ramp_yaw) * self.RAMP_EXIT_OFFSET,
                        self.course_state.ramp_coord.y
                        + math.sin(self.course_state.ramp_yaw) * self.RAMP_EXIT_OFFSET,
                    ),
                    self.course_state.ramp_yaw,
                )
                self.step_state.target = aux.DogModelTarget.TILE
            case aux.DogModelAction.FINISHED:
                self.step_state.remaining_steps = 0
                self.step_state.done = True

        mujoco.mj_step(self.scene, self.data)
        if action in (aux.DogModelAction.TUNNEL, aux.DogModelAction.RAMP):
            self.clear_go2_velocity()

        self.cam_renderer.update_scene(
            self.data,
            camera="go2_camera",
        )

        self.step_state.fb = self.cam_renderer.render()
        self.step_state.last_action = action

        return self.step_state
