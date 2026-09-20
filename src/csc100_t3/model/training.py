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
LOOKED_AROUND = 0.1
APPROACH_DISTANCE = 0.5
APPROACH_REWARD_SCALE = 4.0
BAD_SPECIAL_ACTION = 5

NUM_EPISODES = 500
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_INTERVAL = 10
SAVE_INTERVAL = 10


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
) -> float:
    reward = 0

    # reward when finished near finish tile, punishement when far from tile
    if action == aux.DogModelAction.FINISHED:
        d = distance(state.dog_pos.coord, course.finish_tile_coord)

        reward = MAX_FINISH_REWARD * max(
            -1.0,
            1.0 - d / FINISH_REWARD_RADIUS,
        )

        if state.target != aux.DogModelTarget.TILE:
            reward -= BAD_SPECIAL_ACTION

    # looked somewhere new
    idx = yaw_to_look_idx(sim_mov_r, next_state.dog_pos.yaw)

    if looked[idx] == False:
        looked[idx] = True
        reward += LOOKED_AROUND

    if action == aux.DogModelAction.TUNNEL:
        if state.target != aux.DogModelAction.TUNNEL:
            reward -= BAD_SPECIAL_ACTION

    if action == aux.DogModelAction.RAMP:
        if state.target != aux.DogModelAction.RAMP:
            reward -= BAD_SPECIAL_ACTION

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
    # ramped near ramp (scale with distance)
    # go2's yaw and tunnel's yaw were close when tunnel was trigged
    # go2's yaw and ramp's yaw were close when ramp was trigged
    # reduce reward if collided with obstacle
    return reward


def train_step(
    online_model,
    target_model,
    replay_buffer,
    optimiser,
    gamma: float,
):
    if len(replay_buffer) < replay_buffer.batch_size:
        return

    batch = replay_buffer.sample()

    losses = []

    for transition in batch:
        q_values = online_model(
            transition.state.fb,
            transition.state.target,
        )

        action_index = list(aux.DogModelAction).index(transition.action)

        predicted_q = q_values[action_index]

        with torch.no_grad():
            if transition.next_state.done:
                target_q = torch.tensor(
                    transition.reward,
                    dtype=predicted_q.dtype,
                    device=predicted_q.device,
                )
            else:
                next_online_q = online_model(
                    transition.next_state.fb,
                    transition.next_state.target,
                )

                best_next_action = next_online_q.argmax()

                next_target_q = target_model(
                    transition.next_state.fb,
                    transition.next_state.target,
                )

                future_q = next_target_q[best_next_action]

                target_q = transition.reward + gamma * future_q

        losses.append(
            torch.nn.functional.smooth_l1_loss(
                predicted_q,
                target_q,
            )
        )

    loss = torch.stack(losses).mean()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss.item()


def save_checkpoint(
    model_dir: Path,
    episode: int,
    online_model,
    target_model,
    optimiser,
    epsilon: float,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    path = model_dir / f"checkpoint_{episode:04d}.pt"

    torch.save(
        {
            "episode": episode,
            "online_model": online_model.state_dict(),
            "target_model": target_model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "epsilon": epsilon,
        },
        path,
    )


def run_new(model_dir: Path):
    replay_buffer = ReplayBuffer(
        storage=ListStorage(max_size=50_000),
        batch_size=64,
        collate_fn=lambda x: x,
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

    looked = [False] * int((2 * math.pi) / sim.MOV_R)
    previous_action: aux.DogModelAction | None = None

    for episode in range(NUM_EPISODES):
        state, course = sim.reset(episode)
        looked = [False] * int((2 * math.pi) / sim.MOV_R)
        episode_reward = 0.0

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
                sim.MOV_R,
                previous_action,
            )

            previous_action = action

            replay_buffer.add(
                Transition(
                    state=old_state,
                    course=copy.deepcopy(course),
                    action=action,
                    reward=reward,
                    next_state=copy.deepcopy(next_state),
                )
            )

            loss = train_step(
                online_model,
                target_model,
                replay_buffer,
                optimiser,
                gamma,
            )

            episode_reward += reward
            state = next_state

        epsilon = max(
            MIN_EPSILON,
            epsilon * EPSILON_DECAY,
        )

        if episode % TARGET_UPDATE_INTERVAL == 0:
            target_model.load_state_dict(online_model.state_dict())

        print(episode, episode_reward, epsilon, loss, flush=True)

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(
                model_dir,
                episode + 1,
                online_model,
                target_model,
                optimiser,
                epsilon,
            )
