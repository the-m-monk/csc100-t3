import argparse


def main():
    parser = argparse.ArgumentParser(prog="csc100_t3")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("emptybox")
    sub.add_parser("sceneformat")
    sub.add_parser("model_import_test")

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
