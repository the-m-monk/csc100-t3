import math

from csc100_t3.mjsim import tinterface as ti
from csc100_t3.model import aux


MAX_FINISH_REWARD = 30.0
LOOKED_AROUND = 0.10
LOOK_BIN_SIZE = math.radians(20)
OBSTACLE_CLEARANCE = 0.35
APPROACH_REWARD_SCALE = 7.0
RETREAT_PENALTY_SCALE = 10.0
RETREAT_DEADBAND = 0.005
COLLISION = -15
RAMP_FALL = -20
RAMP_MISSED = -30
TIMEOUT = -30
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


def obstacle_point(
    origin: ti.Vec2,
    yaw: float,
    local_x: float,
    local_y: float,
) -> ti.Vec2:
    return ti.TrainingSimulator.local_to_world(
        origin,
        yaw,
        ti.Vec2(local_x, local_y),
    )


def get_navigation_point(
    state: ti.StepState,
    course: ti.CourseState,
) -> ti.Vec2:
    match state.target:
        case aux.DogModelTarget.TUNNEL:
            local_x = (
                ti.TUNNEL_EXIT_X + OBSTACLE_CLEARANCE
                if state.tunnel_entered
                else ti.TUNNEL_ENTRY_X
            )
            return obstacle_point(
                course.tunnel_coord,
                course.tunnel_yaw,
                local_x,
                ti.TUNNEL_CENTER_Y,
            )

        case aux.DogModelTarget.RAMP:
            local_x = (
                ti.RAMP_EXIT_X + OBSTACLE_CLEARANCE
                if state.ramp_entered
                else ti.RAMP_ENTRY_X
            )
            return obstacle_point(
                course.ramp_coord,
                course.ramp_yaw,
                local_x,
                ti.RAMP_CENTER_Y,
            )

        case aux.DogModelTarget.TILE:
            return course.finish_tile_coord

        case _:
            raise ValueError(f"unknown target: {state.target}")


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
    previous_action: aux.DogModelAction | None,
) -> float:
    reward = 0.0

    if (
        state.target == aux.DogModelTarget.TUNNEL
        and next_state.target == aux.DogModelTarget.RAMP
    ):
        reward += MAX_FINISH_REWARD
        reward += special_yaw_reward(
            next_state.dog_pos.yaw,
            course.tunnel_yaw,
        )

    if (
        state.target == aux.DogModelTarget.RAMP
        and next_state.target == aux.DogModelTarget.TILE
    ):
        reward += MAX_FINISH_REWARD
        reward += special_yaw_reward(
            next_state.dog_pos.yaw,
            course.ramp_yaw,
        )

    if next_state.course_completed:
        reward += MAX_FINISH_REWARD

    idx = yaw_to_look_idx(next_state.dog_pos.yaw)

    if not looked[idx]:
        looked[idx] = True
        reward += LOOKED_AROUND

    navigation_state = next_state if state.target == next_state.target else state
    navigation_point = get_navigation_point(navigation_state, course)

    old_distance = distance(
        state.dog_pos.coord,
        navigation_point,
    )

    new_distance = distance(
        next_state.dog_pos.coord,
        navigation_point,
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

    if state.target == next_state.target:
        old_angle = abs(
            angle_to_trigger(
                state.dog_pos,
                navigation_point,
            )
        )

        new_angle = abs(
            angle_to_trigger(
                next_state.dog_pos,
                navigation_point,
            )
        )

        reward += (old_angle - new_angle) * TURN_REWARD_SCALE

    if next_state.collided_with:
        print(f"COLLISION: {', '.join(next_state.collided_with)}")
        reward += COLLISION

    if next_state.ramp_fell:
        print("FELL_OFF_RAMP")
        reward += RAMP_FALL

    if next_state.ramp_missed:
        print("MISSED_RAMP")
        reward += RAMP_MISSED

    if next_state.done and not next_state.course_completed:
        if state.target == aux.DogModelTarget.RAMP and not state.ramp_summited:
            reward += RAMP_MISSED
        else:
            reward += TIMEOUT

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
