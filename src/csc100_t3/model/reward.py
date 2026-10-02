import math

from csc100_t3.mjsim import tinterface as ti
from csc100_t3.model import aux

TARGET_FRONT_OFFSET = {
    aux.DogModelTarget.TUNNEL: 0.0,
    aux.DogModelTarget.RAMP: 1.1 / 2,
    aux.DogModelTarget.TILE: 0.3 / 2,
}

COMPLETION_ZONE_WIDTH = 1.0
COMPLETION_ZONE_DEPTH = 0.5

PROGRESS_DEADBAND = 0.005

APPROACH_REWARD_SCALE = 7.0
RETREAT_PENALTY_SCALE = 10.0

EARLY_COMPLETION_PENALTY = -5.0
COMPLETION_REWARD = 15.0
ALIGNMENT_BONUS = 2.0

TURN_REWARD_SCALE = 5.0

COLLISION_PENALTY = -10.0

OSCILLATION_PENALTY = -0.25

def distance(a: ti.Vec2, b: ti.Vec2) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def target_pose(
    target: aux.DogModelTarget,
    course: ti.CourseState,
) -> tuple[ti.Vec2, float]:
    match target:
        case aux.DogModelTarget.TUNNEL:
            return course.tunnel_coord, course.tunnel_yaw
        case aux.DogModelTarget.RAMP:
            return course.ramp_coord, course.ramp_yaw
        case aux.DogModelTarget.TILE:
            return course.finish_tile_coord, course.finish_tile_yaw
        case _:
            raise ValueError(f"target has no magi reward: {target}")


def completion_zone_centre(
    target: aux.DogModelTarget,
    course: ti.CourseState,
) -> tuple[ti.Vec2, float]:
    coord, yaw = target_pose(target, course)
    offset = TARGET_FRONT_OFFSET[target] + COMPLETION_ZONE_DEPTH / 2
    return (
        ti.Vec2(
            coord.x - math.cos(yaw) * offset,
            coord.y - math.sin(yaw) * offset,
        ),
        yaw,
    )


def distance_reward(
    state: ti.StepState, next_state: ti.StepState, course: ti.CourseState
):
    r = 0.0

    target_point, _ = completion_zone_centre(state.target, course)

    old_distance = distance(state.dog_pos.coord, target_point)
    new_distance = distance(next_state.dog_pos.coord, target_point)
    progress = old_distance - new_distance

    if progress > PROGRESS_DEADBAND:
        r += progress * APPROACH_REWARD_SCALE
    elif progress < -PROGRESS_DEADBAND:
        r += progress * RETREAT_PENALTY_SCALE

    return r


def bearing_reward(state, next_state, course):
    r = 0.0

    target_point, _ = completion_zone_centre(state.target, course)

    old_bearing = math.atan2(
        target_point.y - state.dog_pos.coord.y,
        target_point.x - state.dog_pos.coord.x,
    )
    new_bearing = math.atan2(
        target_point.y - next_state.dog_pos.coord.y,
        target_point.x - next_state.dog_pos.coord.x,
    )
    old_angle = abs(angle_diff(old_bearing, state.dog_pos.yaw))
    new_angle = abs(angle_diff(new_bearing, next_state.dog_pos.yaw))
    reward += (old_angle - new_angle) * TURN_REWARD_SCALE

    return r


def in_completion_zone(
    dog_pos: ti.DogPos,
    target: aux.DogModelTarget,
    course: ti.CourseState,
) -> bool:
    centre, yaw = completion_zone_centre(target, course)
    dx = dog_pos.coord.x - centre.x
    dy = dog_pos.coord.y - centre.y

    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (
        abs(longitudinal) <= COMPLETION_ZONE_DEPTH / 2
        and abs(lateral) <= COMPLETION_ZONE_WIDTH / 2
    )


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def completion_reward(
    state: ti.StepState,
    course: ti.CourseState,
) -> float:
    if not in_completion_zone(state.dog_pos, state.target, course):
        return EARLY_COMPLETION_PENALTY

    _, target_yaw = target_pose(state.target, course)
    alignment = 1.0 - abs(angle_diff(state.dog_pos.yaw, target_yaw)) / math.pi
    return COMPLETION_REWARD + ALIGNMENT_BONUS * max(0.0, alignment)

def collision_penalty(is_navigation_keepout_respected, state, next_state, course):
    if not is_navigation_keepout_respected(
        next_state.dog_pos.coord,
        course,
        state.target,
    ):
        return COLLISION_PENALTY
    return 0.0

def oscillation_penalty(state, action):
    opposites = {
        aux.DogModelAction.FORWARD: aux.DogModelAction.BACKWARD,
        aux.DogModelAction.BACKWARD: aux.DogModelAction.FORWARD,
        aux.DogModelAction.LEFT: aux.DogModelAction.RIGHT,
        aux.DogModelAction.RIGHT: aux.DogModelAction.LEFT,
    }
    if opposites.get(action) == state.last_action:
        reward += OSCILLATION_PENALTY

def calculate_reward(
    state: ti.StepState,
    next_state: ti.StepState,
    course: ti.CourseState,
    action: aux.DogModelAction,
    is_navigation_keepout_respected,
):
    r = 0.0

    magi_idx = aux.TARGET_TO_MAGI[state.target]
    completion_action = aux.MAGI_ACTIONS[magi_idx][-1]
    if action == completion_action:
        return completion_reward(state, course)

    r += distance_reward(state, next_state, course)
    r += bearing_reward(state, next_state, course)
    r += collision_penalty(is_navigation_keepout_respected, state, next_state, course)
    r += oscillation_penalty(state, action)

    return r
