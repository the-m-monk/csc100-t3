from pathlib import Path

import csc100_t3.mjsim.tinterface as ti


def run_new(model_dir: Path):
    x = ti.TrainingSimulator()
    x.reset(0)
