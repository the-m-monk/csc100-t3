import argparse
from pathlib import Path
import math


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


def mp4_output_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".mp4":
        raise argparse.ArgumentTypeError("recording path must end in .mp4")
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError("recording directory must exist")
    return path


def main():
    parser = argparse.ArgumentParser(prog="csc100_t3")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("emptybox")
    sub.add_parser("sceneformat")
    sub.add_parser("model_import_test")

    train_new = sub.add_parser("train_new")
    train_new.add_argument("--path", type=empty_dir, required=True)
    train_new.add_argument(
        "--weights",
        type=Path,
        help="path to the checkpoint weights used to initialise the magi",
    )
    train_new.add_argument(
        "--magi",
        type=int,
        choices=(0, 1, 2),
        help="train only 0=tunnel, 1=ramp, or 2=tile (default: train all)",
    )
    train_new.add_argument(
        "--epsilon",
        type=epsilon_probability,
        default=1.0,
        help="starting exploration probability (default: 1.0)",
    )

    sim_run_model = sub.add_parser("sim_run_model")
    sim_run_model.add_argument("--path", type=Path, required=True)
    sim_run_model.add_argument(
        "--record",
        type=mp4_output_path,
        help="write a headless MP4 of policy input and action values",
    )
    sim_run_model.add_argument(
        "--seed",
        type=int,
        help="course seed (random by default)",
    )

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

            training.run_new(
                args.path,
                weights_path=args.weights,
                train_magi=args.magi,
                starting_epsilon=args.epsilon,
            )

        case "sim_run_model":
            import csc100_t3.mjsim.run_model as run_model

            run_model.run(args.path, record_path=args.record, seed=args.seed)
