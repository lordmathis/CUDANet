from pathlib import Path
import os

import torch

from gen.base_generator import BaseGenerator

class MatMulGenerator(BaseGenerator):
            
    def __init__(self, seed, fixtures_path):
        super().__init__(seed, fixtures_path)

    def generate(self):
        self._generate_mat_vec_mul(self.fixtures_path / "mat_vec_mul")

    def _generate_mat_vec_mul(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        i = 0
        metadata = []
        for (rows, cols) in [(10, 5), (128, 64), (512, 256)]:
            for dtype in ['float32']:

                matrix = torch.randn(rows, cols)
                matrix_save_path = save_path / f"{i}_matrix.bin"
                self.save_tensor(matrix, matrix_save_path)

                vector = torch.randn(cols)
                vector_save_path = save_path / f"{i}_vector.bin"
                self.save_tensor(vector, vector_save_path)

                expected = torch.matmul(matrix, vector)
                expected_save_path = save_path / f"{i}_expected.bin"
                self.save_tensor(expected, expected_save_path)

                metadata.append([dtype, str(rows), str(cols), matrix_save_path, vector_save_path, expected_save_path])
                
                i += 1

        self.save_metadata(metadata, save_path / "metadata.csv")