from pathlib import Path
from dataclasses import dataclass
import copy
import random
import math

from torchrl.data import ReplayBuffer, ListStorage
import torch

import csc100_t3.mjsim.tinterface as ti
from csc100_t3.model import aux
import csc100_t3.model.reward as rew
import csc100_t3.model as model

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


def choose_action(
    model: model.DogModel,
    fb,
    target,
    last_action: aux.DogModelAction | None,
    epsilon: float,
) -> aux.DogModelAction:
    if random.random() < epsilon:
        return random.choice(list(aux.DogModelAction))

    with torch.no_grad():
        q_values = model(fb, target, last_action)

    index = q_values.argmax().item()

    return list(aux.DogModelAction)[index]


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
            transition.state.last_action,
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
                    transition.next_state.last_action,
                )

                best_next_action = next_online_q.argmax()

                next_target_q = target_model(
                    transition.next_state.fb,
                    transition.next_state.target,
                    transition.next_state.last_action,
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
    for episode in range(NUM_EPISODES):
        state, course = sim.reset(episode)
        looked = [False] * int((2 * math.pi) / sim.MOV_R)
        episode_reward = 0.0

        while not state.done:
            action = choose_action(
                online_model,
                state.fb,
                state.target,
                state.last_action,
                epsilon,
            )

            old_state = copy.deepcopy(state)

            next_state = sim.step(action)
            if old_state.target != next_state.target:
                looked = [False] * int((2 * math.pi) / sim.MOV_R)

            reward = rew.calculate_reward(
                old_state,
                next_state,
                course,
                action,
                looked,
                sim.MOV_R,
                old_state.last_action,
                sim.is_keepout_respected,
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
