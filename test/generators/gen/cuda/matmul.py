import os

import torch

from gen.base_generator import BaseGenerator

class MatMulGenerator(BaseGenerator):
            
    def __init__(self, seed, fixtures_path):
        super().__init__(seed, fixtures_path)
        os.makedirs(self.fixtures_path, exist_ok=True)

    def generate(self):
        i = 0
        metadata = []
        for (rows, cols) in [(10, 5), (128, 64), (512, 256)]:
            for dtype in ['float32']:

                matrix = torch.randn(rows, cols)
                matrix_filename = f"{i}_matrix.bin"
                self.save_tensor(matrix, f"{self.fixtures_path}/{matrix_filename}")

                vector = torch.randn(cols)
                vector_filename = f"{i}_vector.bin"
                self.save_tensor(vector, f"{self.fixtures_path}/{vector_filename}")

                expected = torch.matmul(matrix, vector)
                expected_filename = f"{i}_expected.bin"
                self.save_tensor(expected, f"{self.fixtures_path}/{expected_filename}")

                metadata.append([dtype, str(rows), str(cols), matrix_filename, vector_filename, expected_filename])
                
                i += 1

        self.save_metadata(metadata, f"{self.fixtures_path}/metadata.csv")