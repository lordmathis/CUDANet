import argparse
import shutil
import torch

def clean(fixture_path):
    shutil.rmtree(fixture_path)

def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="CUDANet Test fixtures generator"
    )
    parser.add_argument("--clean", action='store_true', help="Clean fixtures directory")
    parser.add_argument("--seed", default=42, help="Seed for torch random")
    parser.add_argument("--fixtures_path", default="../fixtures", help="Path for generated fixtures")
    
    args = parser.parse_args()

    if args.clean:
        clean(args.fixture_path)

    torch.manual_seed(args.seed)
