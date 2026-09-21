import math

from csc100_t3.mjsim import tinterface as ti
from csc100_t3.model import aux


MAX_FINISH_REWARD = 30.0
FINISH_REWARD_RADIUS = 0.5
LOOKED_AROUND = 0.10
LOOK_BIN_SIZE = math.radians(20)
APPROACH_DISTANCE = 0.5
APPROACH_REWARD_SCALE = 7.0
RETREAT_PENALTY_SCALE = 10.0
RETREAT_DEADBAND = 0.005
BAD_SPECIAL_ACTION = -10
COLLISION = -15
REPEATED_MOVEMENT_AND_NOT_APPROACHING_TRIGGER_POINT = -2.0
OSCILLATION_PENALTY = -0.75
TURN_REWARD_SCALE = 25.0  # increased because turn amount decreased
MAX_SPECIAL_YAW_REWARD = 10.0
SPECIAL_YAW_REWARD_RADIUS = math.radians(8)


def distance(a: ti.Vec2, b: ti.Vec2):
    return math.hypot(a.x - b.x, a.y - b.y)


def yaw_to_look_idx(yaw: float):
    yaw %= 2 * math.pi
    return int(yaw / LOOK_BIN_SIZE)


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def get_trigger_point(
    target: aux.DogModelTarget,
    course: ti.CourseState,
) -> ti.Vec2:
    match target:
        case aux.DogModelTarget.TUNNEL:
            coord = course.tunnel_coord
            yaw = course.tunnel_yaw

        case aux.DogModelTarget.RAMP:
            coord = course.ramp_coord
            yaw = course.ramp_yaw

        case aux.DogModelTarget.TILE:
            coord = course.finish_tile_coord
            yaw = course.finish_tile_yaw

        case _:
            raise ValueError(f"unknown target: {target}")

    return ti.Vec2(
        coord.x - math.cos(yaw) * APPROACH_DISTANCE,
        coord.y - math.sin(yaw) * APPROACH_DISTANCE,
    )


def angle_to_trigger(
    dog_pos: ti.DogPos,
    trigger_point: ti.Vec2,
):
    bearing = math.atan2(
        trigger_point.y - dog_pos.coord.y,
        trigger_point.x - dog_pos.coord.x,
    )

    return angle_diff(bearing, dog_pos.yaw)


def special_yaw_reward(
    dog_yaw: float,
    target_yaw: float,
):
    error = abs(angle_diff(dog_yaw, target_yaw))

    return MAX_SPECIAL_YAW_REWARD * max(
        0.0,
        1.0 - error / SPECIAL_YAW_REWARD_RADIUS,
    )


def calculate_reward(
    state: ti.StepState,
    next_state: ti.StepState,
    course: ti.CourseState,
    action: aux.DogModelAction,
    looked,
    sim_mov_r,
    previous_action: aux.DogModelAction | None,
    ikr,
) -> float:
    reward = 0.0

    if action == aux.DogModelAction.FINISHED:
        d = distance(
            state.dog_pos.coord,
            course.finish_tile_coord,
        )

        reward += MAX_FINISH_REWARD * max(
            -1.0,
            1.0 - d / FINISH_REWARD_RADIUS,
        )

        if state.target != aux.DogModelTarget.TILE:
            reward += BAD_SPECIAL_ACTION

    idx = yaw_to_look_idx(next_state.dog_pos.yaw)

    if not looked[idx]:
        looked[idx] = True
        reward += LOOKED_AROUND

    trigger_point = get_trigger_point(
        state.target,
        course,
    )

    old_distance = distance(
        state.dog_pos.coord,
        trigger_point,
    )

    new_distance = distance(
        next_state.dog_pos.coord,
        trigger_point,
    )

    if state.target == next_state.target:
        progress = old_distance - new_distance

        if progress > RETREAT_DEADBAND:
            reward += progress * APPROACH_REWARD_SCALE
        elif progress < -RETREAT_DEADBAND:
            reward += progress * RETREAT_PENALTY_SCALE

    opposites = {
        aux.DogModelAction.FORWARD: aux.DogModelAction.BACKWARD,
        aux.DogModelAction.BACKWARD: aux.DogModelAction.FORWARD,
        aux.DogModelAction.LEFT: aux.DogModelAction.RIGHT,
        aux.DogModelAction.RIGHT: aux.DogModelAction.LEFT,
    }

    if previous_action is not None and opposites.get(action) == previous_action:
        reward += OSCILLATION_PENALTY

    if action == aux.DogModelAction.TUNNEL:
        d = distance(
            state.dog_pos.coord,
            trigger_point,
        )

        if state.target != aux.DogModelTarget.TUNNEL:
            reward += BAD_SPECIAL_ACTION
        else:
            reward += MAX_FINISH_REWARD * max(
                -1.0,
                1.0 - d / FINISH_REWARD_RADIUS,
            )

            reward += special_yaw_reward(
                state.dog_pos.yaw,
                course.tunnel_yaw,
            )

    if action == aux.DogModelAction.RAMP:
        d = distance(
            state.dog_pos.coord,
            trigger_point,
        )

        if state.target != aux.DogModelTarget.RAMP:
            reward += BAD_SPECIAL_ACTION
        else:
            reward += MAX_FINISH_REWARD * max(
                -1.0,
                1.0 - d / FINISH_REWARD_RADIUS,
            )

            reward += special_yaw_reward(
                state.dog_pos.yaw,
                course.ramp_yaw,
            )

    if state.target == next_state.target:
        old_angle = abs(
            angle_to_trigger(
                state.dog_pos,
                trigger_point,
            )
        )

        new_angle = abs(
            angle_to_trigger(
                next_state.dog_pos,
                trigger_point,
            )
        )

        reward += (old_angle - new_angle) * TURN_REWARD_SCALE

    if not ikr(
        next_state.dog_pos.coord,
        course,
        True,
    ):
        print("COLLISION")
        reward += COLLISION

    if (
        action
        in {
            aux.DogModelAction.FORWARD,
            aux.DogModelAction.BACKWARD,
        }
        and previous_action == action
        and state.target == next_state.target
        and new_distance >= old_distance
    ):
        print("REPEATED_MOVEMENT_AND_NOT_APPROACHING_TRIGGER_POINT")

        reward += REPEATED_MOVEMENT_AND_NOT_APPROACHING_TRIGGER_POINT

    return reward
