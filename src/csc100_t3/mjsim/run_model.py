import time
import random
from pathlib import Path

import mujoco
import mujoco.viewer
import torch

import csc100_t3.mjsim.tinterface as ti
import csc100_t3.model as model
from csc100_t3.model import aux

ACTION_HZ = 2
ACTION_INTERVAL = 1.0 / ACTION_HZ


def run(model_path: Path):
    dog_model = model.DogModel()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    if "online_model" in checkpoint:
        dog_model.load_state_dict(checkpoint["online_model"])
    else:
        dog_model.load_state_dict(checkpoint)

    dog_model.eval()

    sim = ti.TrainingSimulator()

    state, course = sim.reset(random.randint(0, 1_000_000))

    with mujoco.viewer.launch_passive(
        sim.scene,
        sim.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(
                sim.scene,
                viewer.cam,
            )

        next_action_time = time.monotonic()

        while viewer.is_running() and not state.done:
            now = time.monotonic()

            if now >= next_action_time:
                with torch.no_grad():
                    q_values = dog_model(
                        state.fb,
                        state.target,
                        state.last_action,
                    )

                action_index = q_values.argmax().item()

                action = list(aux.DogModelAction)[action_index]

                print(f"target={state.target.name} action={action.name}")

                state = sim.step(action)

                next_action_time = now + ACTION_INTERVAL

            mujoco.mj_step(
                sim.scene,
                sim.data,
            )

            viewer.sync()

            time.sleep(sim.scene.opt.timestep)
