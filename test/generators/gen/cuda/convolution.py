import os

import torch
import torch.nn.functional as F

from gen.base_generator import BaseGenerator


class ConvolutionGenerator(BaseGenerator):
    def __init__(self, seed, fixtures_path, dtypes):
        super().__init__(seed, fixtures_path, dtypes)

    def generate(self):
        os.makedirs(self.fixtures_path, exist_ok=True)

        i = 0
        metadata = []

        # (in_channels, out_channels, kernel_size, stride, padding, input_size)
        configs = [
            (1, 1, 3, 1, 0, 5),
            (3, 8, 3, 1, 1, 32),
            (1, 1, 5, 2, 2, 28),
            (8, 16, 3, 2, 1, 16),
        ]

        for i_c, o_c, k, s, p, size in configs:
            for dtype in self.dtypes:
                # PyTorch uses (N, C, H, W)
                # Our kernel uses (C, H, W) for input and (Filters, H, W) for output
                # But the shapes passed to kernel are [H, W, C]

                input_tensor = torch.randn(1, i_c, size, size)
                kernel = torch.randn(o_c, i_c, k, k)
                bias = torch.randn(o_c)

                output = F.conv2d(input_tensor, kernel, bias, stride=s, padding=p)

                # Reshape for saving in our format [C, H, W]
                # torch input (1, i_c, size, size) -> (i_c, size, size)
                save_input = input_tensor.squeeze(0)
                save_kernel = kernel
                save_bias = bias
                # torch output (1, o_c, oH, oW) -> (o_c, oH, oW)
                save_output = output.squeeze(0)

                input_save_path = self.fixtures_path / f"{i}_input.bin"
                kernel_save_path = self.fixtures_path / f"{i}_kernel.bin"
                bias_save_path = self.fixtures_path / f"{i}_bias.bin"
                expected_save_path = self.fixtures_path / f"{i}_expected.bin"

                self.save_tensor(save_input, input_save_path)
                self.save_tensor(save_kernel, kernel_save_path)
                self.save_tensor(save_bias, bias_save_path)
                self.save_tensor(save_output, expected_save_path)

                oH, oW = save_output.shape[1], save_output.shape[2]

                metadata.append(
                    [
                        dtype,
                        str(size),
                        str(size),
                        str(i_c),  # input H, W, C
                        str(p),
                        str(p),  # padding H, W
                        str(k),
                        str(k),  # kernel H, W
                        str(s),
                        str(s),  # stride H, W
                        str(oH),
                        str(oW),
                        str(o_c),  # output H, W, C (Filters)
                        input_save_path,
                        kernel_save_path,
                        bias_save_path,
                        expected_save_path,
                    ]
                )
                i += 1

        self.save_metadata(metadata, self.fixtures_path / "metadata.csv")
