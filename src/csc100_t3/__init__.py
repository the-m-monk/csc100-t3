import argparse
from pathlib import Path


def empty_dir(value: str) -> Path:
    path = Path(value)

    if not path.is_dir():
        raise argparse.ArgumentTypeError("path must be a directory")

    if any(path.iterdir()):
        raise argparse.ArgumentTypeError("directory must be empty")

    return path


def main():
    parser = argparse.ArgumentParser(prog="csc100_t3")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("emptybox")
    sub.add_parser("sceneformat")
    sub.add_parser("model_import_test")

    train_new = sub.add_parser("train_new")
    train_new.add_argument("--path", type=empty_dir, required=True)

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

            training.run_new(args.path)
