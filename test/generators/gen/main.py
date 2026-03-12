import argparse
import shutil

from pathlib import Path

import torch

from gen.cuda.activation import ActivationGenerator
from gen.cuda.matmul import MatMulGenerator
from gen.cuda.pool import PoolGenerator
from gen.cuda.convolution import ConvolutionGenerator

from gen.tensor.tensor import TensorOpsGenerator

from gen.layers.dense import DenseLayerGenerator


def clean(fixtures_path: Path) -> None:
    if fixtures_path.exists():
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
        default=str(Path(__file__).parent / "../../fixtures"),
        help="Root directory for generated fixtures",
    )
    args = parser.parse_args()

    fixtures_path = Path(args.fixtures_path).resolve()
    fixtures_path.mkdir(parents=True, exist_ok=True)

    if args.clean:
        clean(fixtures_path)
        return

    torch.manual_seed(args.seed)
    dtypes = ["float32"]

    generators = [
        MatMulGenerator(fixtures_path=fixtures_path / "matmul", dtypes=dtypes),
        ActivationGenerator(fixtures_path=fixtures_path / "activation", dtypes=dtypes),
        PoolGenerator(fixtures_path=fixtures_path / "pool", dtypes=dtypes),
        ConvolutionGenerator(
            fixtures_path=fixtures_path / "convolution", dtypes=dtypes
        ),
        TensorOpsGenerator(fixtures_path=fixtures_path / "tensor", dtypes=dtypes),
        DenseLayerGenerator(
            fixtures_path=fixtures_path / "layers" / "dense", dtypes=dtypes
        ),
    ]

    for gen in generators:
        name = type(gen).__name__
        print(f"[{name}] generating fixtures -> {gen.fixtures_path}")
        gen.generate()

    print("Done.")


if __name__ == "__main__":
    main()
