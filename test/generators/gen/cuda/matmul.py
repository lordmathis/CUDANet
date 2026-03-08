from pathlib import Path
import os

import torch

from gen.base_generator import BaseGenerator


class MatMulGenerator(BaseGenerator):
    def __init__(self, seed, fixtures_path):
        super().__init__(seed, fixtures_path)

    def generate(self):
        self._generate_mat_vec_mul(self.fixtures_path / "mat_vec_mul")

        # vector vector ops tests
        self._generate_vec_vec_op(
            self.fixtures_path / "vec_vec_add", [5, 512, 1024], torch.add
        )
        self._generate_vec_vec_op(
            self.fixtures_path / "vec_vec_sub", [5, 512, 1024], torch.sub
        )
        self._generate_vec_vec_op(
            self.fixtures_path / "vec_vec_mul", [5, 512, 1024], torch.mul
        )

        # vector scalar ops tests
        self._generate_vec_scalar_op(
            self.fixtures_path / "vec_scalar_sub", [5, 512, 1024], torch.sub
        )
        self._generate_vec_scalar_op(
            self.fixtures_path / "vec_scalar_add", [5, 512, 1024], torch.add
        )

    def _generate_mat_vec_mul(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        i = 0
        metadata = []
        for rows, cols in [(10, 5), (128, 64), (512, 256)]:
            for dtype in ["float32"]:
                matrix = torch.randn(rows, cols)
                matrix_save_path = save_path / f"{i}_matrix.bin"
                self.save_tensor(matrix, matrix_save_path)

                vector = torch.randn(cols)
                vector_save_path = save_path / f"{i}_vector.bin"
                self.save_tensor(vector, vector_save_path)

                expected = torch.matmul(matrix, vector)
                expected_save_path = save_path / f"{i}_expected.bin"
                self.save_tensor(expected, expected_save_path)

                metadata.append(
                    [
                        dtype,
                        str(rows),
                        str(cols),
                        matrix_save_path,
                        vector_save_path,
                        expected_save_path,
                    ]
                )

                i += 1

        self.save_metadata(metadata, save_path / "metadata.csv")

    def _generate_vec_vec_op(self, save_path, sizes, torch_op):
        os.makedirs(save_path, exist_ok=True)

        i = 0
        metadata = []
        for size in sizes:
            for dtype in ["float32"]:
                vec_a = torch.rand(size)
                vec_a_save_path = save_path / f"{i}_vector_a.bin"
                self.save_tensor(vec_a, vec_a_save_path)

                vec_b = torch.rand(size)
                vec_b_save_path = save_path / f"{i}_vector_b.bin"
                self.save_tensor(vec_b, vec_b_save_path)

                expected = torch_op(vec_a, vec_b)
                expected_save_path = save_path / f"{i}_expected.bin"
                self.save_tensor(expected, expected_save_path)

                metadata.append(
                    [
                        dtype,
                        str(size),
                        vec_a_save_path,
                        vec_b_save_path,
                        expected_save_path,
                    ]
                )

                i += 1

        self.save_metadata(metadata, save_path / "metadata.csv")

    def _generate_vec_scalar_op(self, save_path, sizes, torch_op):
        os.makedirs(save_path, exist_ok=True)

        i = 0
        metadata = []
        for size in sizes:
            for dtype in ["float32"]:
                vector = torch.rand(size)
                vector_save_path = save_path / f"{i}_vector.bin"
                self.save_tensor(vector, vector_save_path)

                scalar = torch.rand(1)
                scalar_save_path = save_path / f"{i}_scalar.bin"
                self.save_tensor(scalar, scalar_save_path)

                expected = torch_op(vector, scalar)
                expected_save_path = save_path / f"{i}_expected.bin"
                self.save_tensor(expected, expected_save_path)

                metadata.append(
                    [
                        dtype,
                        str(size),
                        vector_save_path,
                        scalar_save_path,
                        expected_save_path,
                    ]
                )

                i += 1

        self.save_metadata(metadata, save_path / "metadata.csv")