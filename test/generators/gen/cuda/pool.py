import os

import torch
import torch.nn.functional as F

from gen.base_generator import BaseGenerator


class PoolGenerator(BaseGenerator):
    def __init__(self, seed, fixtures_path, dtypes):
        super().__init__(seed, fixtures_path, dtypes)

    def generate(self):
        self._generate_pool_test(self.fixtures_path / "max_pool", F.max_pool2d)
        self._generate_pool_test(self.fixtures_path / "avg_pool", F.avg_pool2d)

    def _generate_pool_test(self, save_path, torch_op):
        os.makedirs(save_path, exist_ok=True)

        i = 0
        metadata = []
        # (C, H, W), kernel, stride, padding
        test_cases = [
            ((3, 32, 32), (2, 2), (2, 2), (0, 0)),
            ((1, 28, 28), (3, 3), (1, 1), (1, 1)),
            ((16, 14, 14), (2, 2), (2, 2), (0, 0)),
            ((8, 7, 7), (3, 3), (2, 2), (1, 1)),
        ]

        for shape, kernel, stride, padding in test_cases:
            for dtype in self.dtypes:
                C, H, W = shape
                kH, kW = kernel
                sH, sW = stride
                pH, pW = padding

                input_tensor = torch.randn(1, C, H, W)
                input_save_path = save_path / f"{i}_input.bin"
                # Save as (C, H, W)
                self.save_tensor(input_tensor.squeeze(0), input_save_path)

                expected = torch_op(
                    input_tensor, kernel_size=kernel, stride=stride, padding=padding
                )
                expected_save_path = save_path / f"{i}_expected.bin"
                # Save as (C, outH, outW)
                self.save_tensor(expected.squeeze(0), expected_save_path)

                outH, outW = expected.shape[2], expected.shape[3]

                metadata.append(
                    [
                        dtype,
                        str(H),
                        str(W),
                        str(C),
                        str(outH),
                        str(outW),
                        str(kH),
                        str(kW),
                        str(sH),
                        str(sW),
                        str(pH),
                        str(pW),
                        str(input_save_path),
                        str(expected_save_path),
                    ]
                )
                i += 1

        self.save_metadata(metadata, save_path / "metadata.csv")
