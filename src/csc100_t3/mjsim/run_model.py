import random
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch

import csc100_t3.mjsim.tinterface as ti
import csc100_t3.model as model
from csc100_t3.model import aux

ACTION_HZ = 2
ACTION_INTERVAL = 1.0 / ACTION_HZ
RECORDING_SCALE = 8
RECORDING_PANEL_WIDTH = 300


def choose_action(
    dog_model: model.DogModel,
    state: ti.StepState,
) -> tuple[int, torch.Tensor, aux.DogModelAction]:
    magi_idx = aux.TARGET_TO_MAGI[state.target]
    with torch.no_grad():
        q_values = dog_model(
            magi_idx,
            state.fb,
            state.last_action,
            aux.relative_heading_bin(state.dog_pos.yaw, state.start_yaw),
        )

    action_index = q_values.argmax().item()
    return magi_idx, q_values, aux.MAGI_ACTIONS[magi_idx][action_index]


def recording_frame(
    state: ti.StepState,
    magi_idx: int,
    q_values: torch.Tensor,
) -> np.ndarray:
    height, width = state.fb.shape[:2]
    display_size = (width * RECORDING_SCALE, height * RECORDING_SCALE)
    frame = np.zeros(
        (display_size[1], display_size[0] + RECORDING_PANEL_WIDTH, 3),
        dtype=np.uint8,
    )
    frame[:, : display_size[0]] = cv2.cvtColor(
        cv2.resize(state.fb, display_size, interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_RGB2BGR,
    )
    values = q_values.detach().cpu().numpy()
    max_strength = max(float(np.abs(values).max()), 1.0)
    for index, (candidate, value) in enumerate(zip(aux.MAGI_ACTIONS[magi_idx], values)):
        x, y = display_size[0] + 15, 35 + index * 75
        cv2.putText(
            frame,
            f"{candidate.name}: {value:+.2f}",
            (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            1.2,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (x, y + 10), (x + 200, y + 30), (255, 255, 255), 1)
        cv2.rectangle(
            frame,
            (x, y + 10),
            (x + int(200 * abs(float(value)) / max_strength), y + 30),
            (255, 255, 255),
            -1,
        )

    return frame


def record_run(
    dog_model: model.DogModel,
    sim: ti.TrainingSimulator,
    seed: int,
    record_path: Path,
):
    state, _ = sim.reset(seed)
    height, width = state.fb.shape[:2]
    frame_size = (
        width * RECORDING_SCALE + RECORDING_PANEL_WIDTH,
        height * RECORDING_SCALE,
    )
    writer = cv2.VideoWriter(
        str(record_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        ACTION_HZ,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer: {record_path}")

    try:
        while not state.done:
            magi_idx, q_values, action = choose_action(dog_model, state)
            writer.write(recording_frame(state, magi_idx, q_values))
            state = sim.step(action)
    finally:
        writer.release()


def interactive_run(
    dog_model: model.DogModel,
    sim: ti.TrainingSimulator,
    seed: int,
):
    state, _ = sim.reset(seed)

    with mujoco.viewer.launch_passive(
        sim.scene,
        sim.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(sim.scene, viewer.cam)

        next_action_time = time.monotonic()
        while viewer.is_running() and not state.done:
            now = time.monotonic()
            if now >= next_action_time:
                magi_idx, _, action = choose_action(dog_model, state)
                print(
                    f"magi={magi_idx} target={state.target.name} action={action.name}"
                )
                state = sim.step(action)
                next_action_time = now + ACTION_INTERVAL

            mujoco.mj_step(sim.scene, sim.data)
            viewer.sync()
            time.sleep(sim.scene.opt.timestep)


def run(
    model_path: Path,
    record_path: Path | None = None,
    seed: int | None = None,
):
    dog_model = model.DogModel()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = (
        checkpoint["online_model"] if "online_model" in checkpoint else checkpoint
    )
    dog_model.load_state_dict(state_dict)
    dog_model.eval()

    sim = ti.TrainingSimulator()
    course_seed = random.randint(0, 1_000_000) if seed is None else seed
    try:
        if record_path is None:
            interactive_run(dog_model, sim, course_seed)
        else:
            record_run(dog_model, sim, course_seed, record_path)
    finally:
        sim.close()
