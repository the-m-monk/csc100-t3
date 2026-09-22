from pathlib import Path
from dataclasses import dataclass
import copy
import random
import math

import numpy as np
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
ACTION_INDEX = {action: index for index, action in enumerate(aux.DogModelAction)}


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


def batch_inputs(states: list[ti.StepState]):
    frames = np.stack([state.fb for state in states])
    targets = torch.tensor(
        [state.target.value for state in states],
        dtype=torch.long,
    )
    last_action_indices = torch.tensor(
        [
            0 if state.last_action is None else ACTION_INDEX[state.last_action] + 1
            for state in states
        ],
        dtype=torch.long,
    )
    return frames, targets, last_action_indices


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
    frames, targets, last_action_indices = batch_inputs(
        [transition.state for transition in batch]
    )
    q_values = online_model.forward_batch(frames, targets, last_action_indices)
    action_indices = torch.tensor(
        [ACTION_INDEX[transition.action] for transition in batch],
        dtype=torch.long,
    )
    predicted_q = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        target_q = torch.tensor(
            [transition.reward for transition in batch],
            dtype=predicted_q.dtype,
        )
        nonterminal_indices = [
            index for index, transition in enumerate(batch) if not transition.next_state.done
        ]

        if nonterminal_indices:
            next_frames, next_targets, next_last_actions = batch_inputs(
                [batch[index].next_state for index in nonterminal_indices]
            )
            next_online_q = online_model.forward_batch(
                next_frames, next_targets, next_last_actions
            )
            best_next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = target_model.forward_batch(
                next_frames, next_targets, next_last_actions
            )
            future_q = next_target_q.gather(1, best_next_actions).squeeze(1)
            target_q[nonterminal_indices] += gamma * future_q

    loss = torch.nn.functional.smooth_l1_loss(predicted_q, target_q)

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
                old_state.last_action,
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
