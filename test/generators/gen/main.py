import argparse
import os
import shutil

import torch

from gen.cuda.activation import ActivationGenerator
from gen.cuda.matmul import MatMulGenerator
from gen.cuda.pool import PoolGenerator
from gen.cuda.convolution import ConvolutionGenerator

from gen.tensor.tensor import TensorOpsGenerator


def clean(fixtures_path: str) -> None:
    if os.path.exists(fixtures_path):
        shutil.rmtree(fixtures_path)
        print(f"Removed {fixtures_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDANet test fixture generator")
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
    dtypes = ["float32"]

    generators = [
        MatMulGenerator(
            fixtures_path=os.path.join(fixtures_path, "matmul"), dtypes=dtypes
        ),
        ActivationGenerator(
            fixtures_path=os.path.join(fixtures_path, "activation"), dtypes=dtypes
        ),
        PoolGenerator(fixtures_path=os.path.join(fixtures_path, "pool"), dtypes=dtypes),
        ConvolutionGenerator(
            fixtures_path=os.path.join(fixtures_path, "convolution"), dtypes=dtypes
        ),
        TensorOpsGenerator(
            fixtures_path=os.path.join(fixtures_path, "tensor"), dtypes=dtypes
        ),
    ]

    for gen in generators:
        name = type(gen).__name__
        print(f"[{name}] generating fixtures -> {gen.fixtures_path}")
        gen.generate()

    print("Done.")


if __name__ == "__main__":
    main()
