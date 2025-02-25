from __future__ import annotations
from torchvision.models.convnext import ConvNeXt, CNBlockConfig, nn
import torch
from typing import Any, Callable, List, Optional


class ConvNeXt4Depth(ConvNeXt):
    def __init__(
        self,
        in_channels: int,
        block_setting: List[CNBlockConfig],
        **kwargs: Any,
    ) -> None:
        super().__init__(block_setting, **kwargs)
        self.features[0][0] = nn.Conv2d(
            in_channels=in_channels, out_channels=block_setting[0].input_channels, kernel_size=4, stride=4, padding=0, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    @staticmethod
    def build_tiny(in_channels: int = 1, **kwargs: Any) -> ConvNeXt4Depth:
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
        return ConvNeXt4Depth(in_channels, block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)
