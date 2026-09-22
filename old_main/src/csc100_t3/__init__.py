import argparse
import math
from pathlib import Path


def empty_dir(value: str) -> Path:
    path = Path(value)

    if not path.is_dir():
        raise argparse.ArgumentTypeError("path must be a directory")

    if any(path.iterdir()):
        raise argparse.ArgumentTypeError("directory must be empty")

    return path


def epsilon_probability(value: str) -> float:
    try:
        epsilon = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "epsilon must be a number between 0 and 1"
        ) from error

    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise argparse.ArgumentTypeError("epsilon must be a number between 0 and 1")

    return epsilon


def main():
    parser = argparse.ArgumentParser(prog="csc100_t3")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("emptybox")
    sub.add_parser("sceneformat")
    sub.add_parser("model_import_test")

    train_new = sub.add_parser("train_new")
    train_new.add_argument("--path", type=empty_dir, required=True)
    train_new.add_argument("--weights", type=Path)
    train_new.add_argument("--epsilon", type=epsilon_probability, default=1.0)

    sim_run_model = sub.add_parser("sim_run_model")
    sim_run_model.add_argument("--path", type=Path, required=True)

    args = parser.parse_args()

    match args.command:
        case "emptybox":
            from csc100_t3.mjsim.emptybox import run

            run()

        case "sceneformat":
            from csc100_t3.sceneformat import run

            run()

        case "model_import_test":
            from csc100_t3.model import DogModel

        case "train_new":
            from csc100_t3.model import training

            training.run_new(args.path, args.weights, args.epsilon)

        case "sim_run_model":
            import csc100_t3.mjsim.run_model as run_model

            run_model.run(args.path)
