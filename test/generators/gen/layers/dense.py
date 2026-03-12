import os

import torch

from gen.base_generator import BaseGenerator


class DenseLayerGenerator(BaseGenerator):
    def __init__(self, fixtures_path, dtypes):
        super().__init__(fixtures_path, dtypes)

    def generate(self):

        sizes = [(10, 5), (128, 32), (512, 512), (1024, 512), (512, 1024)]
        i = 0
        metadata = []

        for size in sizes:
            for dtype in self.dtypes:
                weights = torch.rand(size[1], size[0])
                weights_save_path =  self.fixtures_path / f"{i}_weights.bin"
                self.save_tensor(weights, weights_save_path)

                bias = torch.rand(size[1])
                bias_save_path = self.fixtures_path / f"{i}_bias.bin"
                self.save_tensor(bias, bias_save_path)

                layer = torch.nn.Linear(size[0], size[1], dtype=dtype)
                layer.weight = weights
                layer.bias = bias

                input = torch.rand(size[0])
                input_save_path = self.fixtures_path / f"{i}_input.bin"
                self.save_tensor(input, input_save_path)

                expected = layer(input)
                expected_save_path = self.fixtures_path / f"{i}_expected.bin"
                self.save_tensor(expected, expected_save_path)

                metadata.append(
                    [
                        dtype,
                        str(size[0]),
                        str(size[1]),
                        weights_save_path,
                        bias_save_path,
                        input_save_path,
                        expected_save_path
                    ]
                )
                i += 1

        self.save_metadata(metadata, self.fixtures_path / "metadata.csv")
