from pathlib import Path
from dataclasses import dataclass
import copy
import random

from torchrl.data import ReplayBuffer, ListStorage
import torch

import csc100_t3.mjsim.tinterface as ti
from csc100_t3.model import aux
import csc100_t3.model as model


@dataclass
class Transition:
    state: ti.StepState
    course: ti.CourseState
    action: aux.DogModelAction
    reward: float
    next_state: ti.StepState


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


def calculate_reward(
    state: ti.StepState,
    next_state: ti.StepState,
    course: ti.CourseState,
    action: aux.DogModelAction,
) -> float: ...


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

    while not state.done:
        action = choose_action(
            online_model,
            state.fb,
            state.target,
            epsilon,
        )
    
        old_state = copy.deepcopy(state)
    
        next_state = sim.step(action)
    
        reward = calculate_reward(
            old_state,
            next_state,
            course,
            action,
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
