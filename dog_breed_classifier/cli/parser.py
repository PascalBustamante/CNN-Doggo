import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument(
        "--mode",
        choices=["train", "optimize", "evaluate"],
        default="train",
        help="Select the mode (train, optimize, evaluate)",
    )
    parser.add_argument(
        "--other_param",
        type=int,
        default=42,
        help="Any other parameter you might want to set",
    )

    return parser.parse_args()
