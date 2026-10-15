import os

import torch
import torch.nn.functional as F

from gen.base_generator import BaseGenerator, to_torch_dtype


class Conv2dLayerGenerator(BaseGenerator):
    def __init__(self, fixtures_path, dtypes):
        super().__init__(fixtures_path, dtypes)

    def generate(self):
        os.makedirs(self.fixtures_path, exist_ok=True)

        # (in_channels, out_channels, kernel_size, stride, padding, input_size)
        configs = [
            (1, 1, 3, 1, 0, 5),
            (3, 8, 3, 1, 1, 32),
            (1, 1, 5, 2, 2, 28),
            (8, 16, 3, 2, 1, 16),
        ]

        i = 0
        metadata = []

        for i_c, o_c, k, s, p, size in configs:
            for dtype in self.dtypes:
                torch_dtype = to_torch_dtype(dtype)

                # PyTorch uses (N, C, H, W), CUDANet kernel consumes
                # input (C, H, W), weights (F, C, kH, kW) and
                # output (F, oH, oW)
                input = torch.randn(1, i_c, size, size, dtype=torch_dtype)
                weights = torch.randn(o_c, i_c, k, k, dtype=torch_dtype)
                bias = torch.randn(o_c, dtype=torch_dtype)

                expected = F.conv2d(input, weights, bias, stride=s, padding=p)

                weights_save_path = self.fixtures_path / f"{i}_weights.bin"
                self.save_tensor(weights, weights_save_path)

                bias_save_path = self.fixtures_path / f"{i}_bias.bin"
                self.save_tensor(bias, bias_save_path)

                input_save_path = self.fixtures_path / f"{i}_input.bin"
                self.save_tensor(input.squeeze(0), input_save_path)

                expected_save_path = self.fixtures_path / f"{i}_expected.bin"
                self.save_tensor(expected.squeeze(0), expected_save_path)

                metadata.append(
                    [
                        dtype,
                        str(size),
                        str(size),
                        str(i_c),  # input H, W, C
                        str(k),
                        str(k),
                        str(o_c),  # kernel H, W, filters
                        str(s),
                        str(s),  # stride H, W
                        str(p),
                        str(p),  # padding H, W
                        weights_save_path,
                        bias_save_path,
                        input_save_path,
                        expected_save_path,
                    ]
                )
                i += 1

        self.save_metadata(metadata, self.fixtures_path / "metadata.csv")
