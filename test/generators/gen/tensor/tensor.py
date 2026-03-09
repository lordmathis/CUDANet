import os

import torch

from gen.base_generator import BaseGenerator


class TensorOpsGenerator(BaseGenerator):
    def __init__(self, seed, fixtures_path):
        super().__init__(seed, fixtures_path)

    def generate(self):
        self._generate_tensor_ops(
            self.fixtures_path / "max", [5, 128, 512, 1024], torch.max
        )

    def _generate_tensor_ops(self, save_path, sizes, torch_op):
        os.makedirs(save_path, exist_ok=True)

        i = 0
        metadata = []
        for size in sizes:
            for dtype in ["float32"]:
                vector = torch.randn(size)
                vector_save_path = save_path / f"{i}_vector.bin"
                self.save_tensor(vector, vector_save_path)

                expected = torch_op(vector)
                expected_save_path = save_path / f"{i}_expected.bin"
                self.save_tensor(expected, expected_save_path)

                metadata.append(
                    [
                        dtype,
                        str(size),
                        vector_save_path,
                        expected_save_path,
                    ]
                )
                i += 1

        self.save_metadata(metadata, save_path / "metadata.csv")
