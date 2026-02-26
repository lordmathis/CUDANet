import argparse
import shutil
import os

import torch

from gen.cuda.matmul import MatMulGenerator


def clean(fixtures_path: str) -> None:
    if os.path.exists(fixtures_path):
        shutil.rmtree(fixtures_path)
        print(f"Removed {fixtures_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUDANet test fixture generator"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove fixtures directory and exit"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global random seed (default: 42)"
    )
    parser.add_argument(
        "--fixtures_path",
        default=os.path.join(os.path.dirname(__file__), "../../fixtures"),
        help="Root directory for generated fixtures",
    )
    args = parser.parse_args()

    fixtures_path = os.path.abspath(args.fixtures_path)
    os.makedirs(fixtures_path, exist_ok=True)

    if args.clean:
        clean(fixtures_path)
        return

    torch.manual_seed(args.seed)

    generators = [
        MatMulGenerator(
            seed=args.seed,
            fixtures_path=os.path.join(fixtures_path, "matmul"),
        ),
    ]

    for gen in generators:
        name = type(gen).__name__
        print(f"[{name}] generating fixtures -> {gen.fixtures_path}")
        gen.generate()

    print("Done.")


if __name__ == "__main__":
    main()
