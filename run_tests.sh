#!/bin/bash

set -eou pipefail

# Build library and tests
cmake -DBUILD_TESTS=ON -B build
cmake --build build

# Generate fixturea
cd test/generators
uv run python -m gen.main --clean
uv run python -m gen.main

# Run tests
cd ..
../build/test/test_main

