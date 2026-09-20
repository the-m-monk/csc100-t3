import math

from csc100_t3.mjsim import tinterface as ti
from csc100_t3.model import aux

MAX_FINISH_REWARD = 20  # used by TP for tunnel and ramp too
FINISH_REWARD_RADIUS = 1.0
LOOKED_AROUND = 0.5
APPROACH_DISTANCE = 0.5
APPROACH_REWARD_SCALE = 4.0
BAD_SPECIAL_ACTION = -5
COLLISION = -7
REPEATED_MOVEMENT_AND_MOVING_AWAY_FROM_TRIGGER_POINT = -2


def distance(a: ti.Vec2, b: ti.Vec2):
    return math.hypot(a.x - b.x, a.y - b.y)


def yaw_to_look_idx(sim_mov_r, y):
    y = y % (2 * math.pi)
    return int(y / sim_mov_r) % int((2 * math.pi) / sim_mov_r)


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
    reward = 0

    # reward when finished near finish tile, punishement when far from tile
    if action == aux.DogModelAction.FINISHED:
        d = distance(state.dog_pos.coord, course.finish_tile_coord)

        reward += MAX_FINISH_REWARD * max(
            -1.0,
            1.0 - d / FINISH_REWARD_RADIUS,
        )

        if state.target != aux.DogModelTarget.TILE:
            reward += BAD_SPECIAL_ACTION

    # looked somewhere new
    idx = yaw_to_look_idx(sim_mov_r, next_state.dog_pos.yaw)

    if looked[idx] == False:
        looked[idx] = True
        reward += LOOKED_AROUND

    if action == aux.DogModelAction.TUNNEL:
        if state.target != aux.DogModelAction.TUNNEL:
            reward += BAD_SPECIAL_ACTION

    if action == aux.DogModelAction.RAMP:
        if state.target != aux.DogModelAction.RAMP:
            reward += BAD_SPECIAL_ACTION

    # went towards target
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

        reward += progress * APPROACH_REWARD_SCALE

    # punish oscillation
    opposites = {
        aux.DogModelAction.FORWARD: aux.DogModelAction.BACKWARD,
        aux.DogModelAction.BACKWARD: aux.DogModelAction.FORWARD,
        aux.DogModelAction.LEFT: aux.DogModelAction.RIGHT,
        aux.DogModelAction.RIGHT: aux.DogModelAction.LEFT,
    }

    if previous_action is not None and opposites.get(action) == previous_action:
        reward -= 0.2

    # tunnelled near tunnel spot (scale with distance)
    if action == aux.DogModelAction.TUNNEL:
        d = distance(state.dog_pos.coord, trigger_point)

        reward += MAX_FINISH_REWARD * max(
            -1.0,
            1.0 - d / FINISH_REWARD_RADIUS,
        )

        if state.target != aux.DogModelTarget.TUNNEL:
            reward += BAD_SPECIAL_ACTION

    # ramped near ramp (scale with distance)
    if action == aux.DogModelAction.RAMP:
        d = distance(state.dog_pos.coord, trigger_point)

        reward += MAX_FINISH_REWARD * max(
            -1.0,
            1.0 - d / FINISH_REWARD_RADIUS,
        )

        if state.target != aux.DogModelTarget.RAMP:
            reward += BAD_SPECIAL_ACTION

    # go2's yaw and tunnel's yaw were close when tunnel was trigged
    # go2's yaw and ramp's yaw were close when ramp was trigged

    # reduce reward if collided with obstacle
    if ikr(next_state.dog_pos.coord, course, True) == False:
        print("COLLISION")
        reward += COLLISION

    # punish repeated movement without getting closer to target
    if (
        previous_action == action
        and state.target == next_state.target
        and (
            distance(state.dog_pos.coord, trigger_point)
            < distance(next_state.dog_pos.coord, trigger_point)
        )
    ):
        reward += REPEATED_MOVEMENT_AND_MOVING_AWAY_FROM_TRIGGER_POINT

    return reward
