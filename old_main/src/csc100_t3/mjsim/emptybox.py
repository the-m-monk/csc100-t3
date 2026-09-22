import time
import random

import mujoco
import mujoco.viewer

import csc100_t3.mjsim.tinterface as ti


def run():
    x = ti.TrainingSimulator()

    x.reset(random.randint(0, 1_000_000))

    with mujoco.viewer.launch_passive(
        x.scene,
        x.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(x.scene, viewer.cam)

        while viewer.is_running():
            mujoco.mj_step(x.scene, x.data)

            viewer.sync()

            time.sleep(x.scene.opt.timestep)
