from pathlib import Path
from dataclasses import dataclass
import copy
import random
import math

from torchrl.data import ReplayBuffer, ListStorage
import torch

import csc100_t3.mjsim.tinterface as ti
from csc100_t3.model import aux
import csc100_t3.model as model

MAX_FINISH_REWARD = 20
FINISH_REWARD_RADIUS = 1.0
LOOKED_AROUND = 3

@dataclass
class Transition:
    state: ti.StepState
    course: ti.CourseState
    action: aux.DogModelAction
    reward: float
    next_state: ti.StepState

def distance(a: ti.Vec2, b: ti.Vec2):
    return math.hypot(a.x - b.x, a.y - b.y)

def choose_action(
    model: model.DogModel,
    fb,
    target,
    epsilon: float,
) -> aux.DogModelAction:
    if random.random() < epsilon:
        return random.choice(list(aux.DogModelAction))

    with torch.no_grad():
        q_values = model(fb, target)

    index = q_values.argmax().item()

    return list(aux.DogModelAction)[index]

def yaw_to_look_idx(sim_mov_r, y):
    y = y % (2 * math.pi)
    return int(y / sim_mov_r) % int((2 * math.pi) / sim_mov_r)

def calculate_reward(
    state: ti.StepState,
    next_state: ti.StepState,
    course: ti.CourseState,
    action: aux.DogModelAction,
    looked,
    sim_mov_r,
) -> float:
    reward = 0
    
    # finished near finish tile 
    if action == aux.DogModelAction.FINISHED:
        d = distance(state.dog_pos.coord, course.finish_tile_coord)

        reward = MAX_FINISH_REWARD * max(
            -1.0,
            1.0 - d / FINISH_REWARD_RADIUS,
        )

    # looked somewhere new
    if looked[
        yaw_to_look_idx(sim_mov_r, state.dog_pos.yaw)
    ] == False:
        reward += LOOKED_AROUND

    # punish if finished an target is not tile
    # punish if ramped when target is not ramp
    # punish if tunelled when target is not tunelled

    # went towards target

    # tunnelled near tunnel spot (scale with distance)
    # ramped near ramp (scale with distance)
    # go2's yaw and tunnel's yaw were close when tunnel was trigged
    # go2's yaw and ramp's yaw were close when ramp was trigged
    # reduce reward if collided with obstacle
    return reward

def run_new(model_dir: Path):
    replay_buffer = ReplayBuffer(
        storage=ListStorage(max_size=50_000),
        batch_size=64,
    )

    online_model = model.DogModel()
    target_model = copy.deepcopy(online_model)
    target_model.eval()
    target_model.requires_grad_(False)

    optimiser = torch.optim.Adam(
        online_model.parameters(),
        lr=1e-4,
    )

    epsilon = 1.0
    gamma = 0.99

    sim = ti.TrainingSimulator()

    state, course = sim.reset(0)

    looked = [False] * int((2 * math.pi) / sim.MOV_R)

    while not state.done:
        action = choose_action(
            online_model,
            state.fb,
            state.target,
            epsilon,
        )
    
        old_state = copy.deepcopy(state)
    
        next_state = sim.step(action)

        if old_state.target != next_state.target:
            looked = [False] * int((2 * math.pi) / sim.MOV_R)
    
        reward = calculate_reward(
            old_state,
            next_state,
            course,
            action,
            looked,
            sim.MOV_R
        )
    
        replay_buffer.add(
            Transition(
                state=old_state,
                course=copy.deepcopy(course),
                action=action,
                reward=reward,
                next_state=copy.deepcopy(next_state),
            )
        )
    
        state = next_state
