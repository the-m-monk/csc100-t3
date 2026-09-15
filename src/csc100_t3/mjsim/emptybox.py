import time
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE_PATH = Path(__file__).resolve().parents[3] / "scene" / "main.xml"

def run():
    scene = mujoco.MjModel.from_xml_path(str(SCENE_PATH))

    state = mujoco.MjData(scene)

    with mujoco.viewer.launch_passive(
        scene,
        state,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(scene, viewer.cam)

        while viewer.is_running():
            mujoco.mj_step(scene, state)

            viewer.sync()

            time.sleep(scene.opt.timestep)
