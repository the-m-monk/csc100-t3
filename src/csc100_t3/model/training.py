from pathlib import Path
from dataclasses import dataclass
import copy
import random

import numpy as np
import torch
from torchrl.data import ListStorage, ReplayBuffer

import csc100_t3.mjsim.tinterface as ti
import csc100_t3.model as model
import csc100_t3.model.reward as rew
from csc100_t3.model import aux

NUM_EPISODES = 500
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_INTERVAL = 10
SAVE_INTERVAL = 10
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GAMMA = 0.99


@dataclass
class Transition:
    state: ti.StepState
    action: aux.DogModelAction
    reward: float
    next_state: ti.StepState


def choose_action(
    dog_model: model.DogModel,
    magi_idx: int,
    state: ti.StepState,
    epsilon: float,
) -> aux.DogModelAction:
    actions = aux.MAGI_ACTIONS[magi_idx]
    if random.random() < epsilon:
        return random.choice(actions)

    with torch.no_grad():
        q_values = dog_model(
            magi_idx,
            state.fb,
            state.last_action,
            aux.relative_heading_bin(state.dog_pos.yaw, state.start_yaw),
        )

    return actions[q_values.argmax().item()]


def batch_inputs(
    magi_idx: int,
    states: list[ti.StepState],
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    actions = aux.MAGI_ACTIONS[magi_idx]
    frames = np.stack([state.fb for state in states])
    last_action_indices = torch.tensor(
        [
            0 if state.last_action is None else actions.index(state.last_action) + 1
            for state in states
        ],
        dtype=torch.long,
    )
    relative_heading_bins = torch.tensor(
        [
            aux.relative_heading_bin(state.dog_pos.yaw, state.start_yaw)
            for state in states
        ],
        dtype=torch.long,
    )
    return frames, last_action_indices, relative_heading_bins


def train_step(
    magi_idx: int,
    online_model: model.DogModel,
    target_model: model.DogModel,
    replay_buffer: ReplayBuffer,
    optimiser: torch.optim.Optimizer,
) -> float | None:
    if len(replay_buffer) < replay_buffer.batch_size:
        return None

    batch = replay_buffer.sample()
    actions = aux.MAGI_ACTIONS[magi_idx]
    frames, last_action_indices, relative_heading_bins = batch_inputs(
        magi_idx,
        [transition.state for transition in batch],
    )
    q_values = online_model.forward_batch(
        magi_idx,
        frames,
        last_action_indices,
        relative_heading_bins,
    )
    action_indices = torch.tensor(
        [actions.index(transition.action) for transition in batch],
        dtype=torch.long,
    )
    predicted_q = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        target_q = torch.tensor(
            [transition.reward for transition in batch],
            dtype=predicted_q.dtype,
        )
        nonterminal_indices = [
            index
            for index, transition in enumerate(batch)
            if not transition.next_state.done
        ]
        if nonterminal_indices:
            next_frames, next_last_actions, next_relative_heading_bins = batch_inputs(
                magi_idx,
                [batch[index].next_state for index in nonterminal_indices],
            )
            next_online_q = online_model.forward_batch(
                magi_idx,
                next_frames,
                next_last_actions,
                next_relative_heading_bins,
            )
            best_next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = target_model.forward_batch(
                magi_idx,
                next_frames,
                next_last_actions,
                next_relative_heading_bins,
            )
            future_q = next_target_q.gather(1, best_next_actions).squeeze(1)
            target_q[nonterminal_indices] += GAMMA * future_q

    loss = torch.nn.functional.smooth_l1_loss(predicted_q, target_q)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    return loss.item()


def save_checkpoint(
    model_dir: Path,
    episode: int,
    online_model: model.DogModel,
    target_model: model.DogModel,
    optimisers: dict[int, torch.optim.Optimizer],
    epsilons: dict[int, float],
):
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"checkpoint_{episode:04d}.pt"
    torch.save(
        {
            "episode": episode,
            "online_model": online_model.state_dict(),
            "target_model": target_model.state_dict(),
            "optimisers": {
                magi_idx: optimiser.state_dict()
                for magi_idx, optimiser in optimisers.items()
            },
            "epsilons": epsilons,
        },
        path,
    )


def load_base_weights(
    online_model: model.DogModel,
    weights_path: Path,
):
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = saved["online_model"] if "online_model" in saved else saved
    online_model.load_state_dict(state_dict)


def run_new(
    model_dir: Path,
    weights_path: Path | None = None,
    train_magi: int | None = None,
    starting_epsilon: float = 1.0,
):
    if train_magi is not None and not 0 <= train_magi < len(aux.MAGI_TARGETS):
        raise ValueError(f"unknown magi index: {train_magi}")

    magi_indices = (
        tuple(range(len(aux.MAGI_TARGETS))) if train_magi is None else (train_magi,)
    )
    online_model = model.DogModel()
    if weights_path is not None:
        load_base_weights(online_model, weights_path)

    for magi_idx, magi in enumerate(online_model.magi):
        magi.requires_grad_(magi_idx in magi_indices)

    target_model = copy.deepcopy(online_model)
    target_model.eval()
    target_model.requires_grad_(False)

    replay_buffers = {
        magi_idx: ReplayBuffer(
            storage=ListStorage(max_size=REPLAY_CAPACITY),
            batch_size=BATCH_SIZE,
            collate_fn=lambda transitions: transitions,
        )
        for magi_idx in magi_indices
    }
    optimisers = {
        magi_idx: torch.optim.Adam(
            online_model.magi[magi_idx].parameters(),
            lr=LEARNING_RATE,
        )
        for magi_idx in magi_indices
    }
    epsilons = {magi_idx: starting_epsilon for magi_idx in magi_indices}

    sim = ti.TrainingSimulator()

    for episode in range(NUM_EPISODES):
        _, generated_course = sim.reset(episode)
        course = copy.deepcopy(generated_course)

        for magi_idx in magi_indices:
            target = aux.MAGI_TARGETS[magi_idx]
            state = sim.reset_subcourse(target)
            episode_reward = 0.0
            loss = None

            while not state.done:
                action = choose_action(
                    online_model,
                    magi_idx,
                    state,
                    epsilons[magi_idx],
                )
                old_state = copy.deepcopy(state)
                next_state = sim.step_subcourse(action)
                reward = rew.calculate_reward(
                    old_state,
                    next_state,
                    course,
                    action,
                    sim.is_navigation_keepout_respected,
                )

                replay_buffers[magi_idx].add(
                    Transition(
                        state=old_state,
                        action=action,
                        reward=reward,
                        next_state=copy.deepcopy(next_state),
                    )
                )
                loss = train_step(
                    magi_idx,
                    online_model,
                    target_model,
                    replay_buffers[magi_idx],
                    optimisers[magi_idx],
                )
                episode_reward += reward
                state = next_state

            epsilons[magi_idx] = max(
                MIN_EPSILON,
                epsilons[magi_idx] * EPSILON_DECAY,
            )
            print(
                f"course={episode:05} magi={magi_idx} target={target.name:<6} "
                f"reward={episode_reward:03.5} "
                f"epsilon={epsilons[magi_idx]:.3f} "
                f"loss={loss}",
                flush=True,
            )

        if (episode + 1) % TARGET_UPDATE_INTERVAL == 0:
            for magi_idx in magi_indices:
                target_model.magi[magi_idx].load_state_dict(
                    online_model.magi[magi_idx].state_dict()
                )

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(
                model_dir,
                episode + 1,
                online_model,
                target_model,
                optimisers,
                epsilons,
            )

    sim.close()
