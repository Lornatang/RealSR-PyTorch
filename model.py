# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch

__all__ = [
    "RealSRRCAN",
    "realsr_rcan_x2", "realsr_rcan_x3", "realsr_rcan_x4",
]


# Copy from `https://github.com/Alan-xw/RealSR/blob/master/model/common.py`.
def _get_x_offset(x: Tensor, q: Tensor, patch_size: int) -> Tensor:
    # Shape: (b,h,w,2N)
    batch_size, height, width, _ = q.shape
    _, channels, _, padded_w = x.shape
    # (b, c, h*padded_w)
    x = x.contiguous().view(batch_size, channels, -1)
    # (b, h, w, N)
    # index_x*w + index_y
    index = torch.add(q[..., :patch_size] * padded_w, q[..., patch_size:])

    # (b, c, h*w*N)
    index = index.contiguous().unsqueeze(dim=1)
    index = index.expand(-1, channels, -1, -1, -1).contiguous().view(batch_size, channels, -1)
    x_offset = x.gather(dim=-1, index=index).contiguous().view(batch_size, channels, height, width, patch_size)

    return x_offset


class RealSRRCAN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            channels: int = 64,
            reduction: int = 16,
            num_rg: int = 10,
            num_rcab: int = 20,
            upscale_factor: int = 4,
    ) -> None:
        super(RealSRRCAN, self).__init__()
        self.upscale_factor = upscale_factor

        # Low frequency information extraction layer
        self.conv1 = nn.Conv2d(in_channels, int(channels // 16), (3, 3), (1, 1), (1, 1))

        # Shuffle down-sampling
        self.shuffle_downsampling = _ShuffleDownSampling(4)

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rg):
            trunk.append(_ResidualGroup(channels, reduction, num_rcab))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        self.laplacian_pyramid = _LaplacianPyramid()
        self.laplacian_reconstruction = _LaplacianReconstruction()

        self.shuffle_up_sampling1 = _ShuffleUpSampling(4)
        self.shuffle_up_sampling2 = _ShuffleUpSampling(2)
        self.shuffle_up_sampling3 = _ShuffleUpSampling(1)

        self.branch1 = _ApplyPerPixelKernels(int(channels // 16), rate=4, kernel_size=5)
        self.branch2 = _ApplyPerPixelKernels(int(channels // 4), rate=2, kernel_size=5)
        self.branch3 = _ApplyPerPixelKernels(int(channels // 1), rate=1, kernel_size=5)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        # Upscale image size
        x = F_torch.interpolate(x, scale_factor=self.upscale_factor, mode="nearest")

        # LaplacianPyramid
        gaussian_lists, laplacian_lists = self.laplacian_pyramid(x)
        laplacian1, laplacian2 = laplacian_lists[0], laplacian_lists[1]
        gaussian3 = gaussian_lists[-1]

        out = self.conv1(x)
        out = self.shuffle_downsampling(out)
        out = self.trunk(out)
        out = self.conv2(out)

        # Branch_1 out, image_size // 16
        shuffle_up_sampling1 = self.shuffle_up_sampling1(out)
        branch1_out = self.branch1(shuffle_up_sampling1, laplacian1)
        # Branch_2 out, image_size // 4
        shuffle_up_sampling2 = self.shuffle_up_sampling2(out)
        branch2_out = self.branch2(shuffle_up_sampling2, laplacian2)
        # Branch_3 out, image_size // 1
        _ = self.shuffle_up_sampling3(out)
        branch3_out = self.branch3(out, gaussian3)

        # LaplacianReconstruction
        rec_x2 = self.laplacian_reconstruction(branch2_out, branch3_out)
        rec_x4 = self.laplacian_reconstruction(branch1_out, rec_x2)

        out = torch.clamp_(rec_x4, 0.0, 1.0)

        return out


class _ShuffleDownSampling(nn.Module):
    def __init__(self, downscale_factor: int) -> None:
        super(_ShuffleDownSampling, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape

        # Reduce input image size
        channels_out = channels * (self.downscale_factor ** 2)
        height_out = height // self.downscale_factor
        width_out = width // self.downscale_factor

        out = x.contiguous().view(batch_size,
                                  channels,
                                  height_out,
                                  self.downscale_factor,
                                  width_out,
                                  self.downscale_factor)
        out = out.contiguous().permute(0, 1, 3, 5, 2, 4)
        out = out.contiguous().view(batch_size, channels_out, height_out, width_out)

        return out


class _ShuffleUpSampling(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(_ShuffleUpSampling, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        out = F_torch.pixel_shuffle(x, self.upscale_factor)

        return out


# Modified from `https://github.com/Alan-xw/RealSR/blob/master/model/common.py`
class _ApplyPerPixelKernels(nn.Module):
    def __init__(self, channels: int, rate: int, kernel_size: int) -> None:
        super(_ApplyPerPixelKernels, self).__init__()
        hidden_channels = channels * rate ** 2
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.zero_padding = nn.ZeroPad2d(self.padding)
        self.kernel_conv = nn.Sequential(*[
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Conv2d(hidden_channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Conv2d(hidden_channels, int(3 * kernel_size ** 2), (3, 3), (1, 1), (1, 1))
        ])

    def forward(self, feature_x: Tensor, x: Tensor) -> Tensor:
        kernel_set = self.kernel_conv(feature_x)

        dtype = kernel_set.data.type()
        patch_size = self.kernel_size ** 2
        # padding the input image with zero values
        if self.padding:
            x = self.zero_padding(x)

        p = self.__get_abs_pixel_index(kernel_set, dtype)
        p = p.contiguous().permute(0, 2, 3, 1).long()
        x_pixel_set = _get_x_offset(x, p, patch_size)
        _, _, height, width = kernel_set.shape
        kernel_set_reshape = kernel_set.reshape(-1, self.kernel_size ** 2, 3, height, width).permute(0, 2, 3, 4, 1)
        x_ = x_pixel_set

        out = torch.mul(x_, kernel_set_reshape)
        out = out.sum(dim=-1, keepdim=True).squeeze(dim=-1)

        return out

    def __get_abs_pixel_index(self, kernel_set: Tensor, dtype: torch.dtype) -> Tensor:
        patch_size = self.kernel_size ** 2
        batch_size, height, width = kernel_set.size(0), kernel_set.size(2), kernel_set.size(3)
        # get absolute index of center index
        p_0_x, p_0_y = np.meshgrid(range(self.padding, height + self.padding),
                                   range(self.padding, width + self.padding),
                                   indexing="ij")
        p_0_x = p_0_x.flatten().reshape(1, 1, height, width).repeat(patch_size, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, height, width).repeat(patch_size, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)
        p_0.requires_grad = False

        # get relative index around center pixel
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   indexing="ij")
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * patch_size, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)
        p_n.requires_grad = False
        p = torch.add(p_0, p_n)
        p = p.repeat(batch_size, 1, 1, 1)

        return p


# Modified from `https://github.com/Alan-xw/RealSR/blob/master/model/common.py`
class _GaussianBlur(nn.Module):
    def __init__(self) -> None:
        super(_GaussianBlur, self).__init__()
        kernel = np.array([[1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [6. / 256., 24. / 256., 36. / 256., 24. / 256., 6. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.]])

        kernel = torch.FloatTensor(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.gaussian = nn.Conv2d(3, 3, (5, 5), (1, 1), (2, 2), groups=3, bias=False)
        self.gaussian.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.gaussian(x)

        return out


# Modified from `https://github.com/Alan-xw/RealSR/blob/master/model/common.py`
class _GaussianBlurUp(nn.Module):
    def __init__(self) -> None:
        super(_GaussianBlurUp, self).__init__()
        kernel = np.array([[1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [6. / 256., 24. / 256., 36. / 256., 24. / 256., 6. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.]])
        kernel = kernel * 4
        kernel = torch.FloatTensor(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.gaussian = nn.Conv2d(3, 3, (5, 5), (1, 1), (2, 2), groups=3, bias=False)
        self.gaussian.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.gaussian(x)

        return out


# Modified from `https://github.com/Alan-xw/RealSR/blob/master/model/common.py`
class _LaplacianPyramid(nn.Module):
    def __init__(self, step: int = 3):
        super(_LaplacianPyramid, self).__init__()
        self.gaussian_blur = _GaussianBlur()
        self.gaussian_blur_up = _GaussianBlurUp()
        self.step = step

    def forward(self, x: Tensor) -> tuple[list[Tensor | Any], list[Any]]:
        gaussian_lists = [x]
        laplacian_lists = []
        size_lists = [x.shape[2:]]

        for _ in range(self.step - 1):
            gaussian_down = self.gaussian_blur(gaussian_lists[-1])
            gaussian_down = gaussian_down[:, :, ::2, ::2]

            gaussian_lists.append(gaussian_down)
            size_lists.append(gaussian_down.shape[2:])

            batch_size, channels, _, _ = gaussian_lists[-1].shape
            height, width = size_lists[-2]
            gaussian_up = torch.zeros((batch_size, channels, height, width), device=x.device)
            gaussian_up[:, :, ::2, ::2] = gaussian_lists[-1]
            gaussian_up = self.gaussian_blur_up(gaussian_up)

            Lap = torch.sub(gaussian_lists[-2], gaussian_up)
            laplacian_lists.append(Lap)

        return gaussian_lists, laplacian_lists


# Modified from `https://github.com/Alan-xw/RealSR/blob/master/model/common.py`
class _LaplacianReconstruction(nn.Module):
    def __init__(self) -> None:
        super(_LaplacianReconstruction, self).__init__()
        self.gaussian_blur_up = _GaussianBlurUp()

    def forward(self, x_laplacian: Tensor, x_gaussian: Tensor) -> Tensor:
        batch_size, channels, height, width = x_gaussian.shape
        out = torch.zeros((batch_size, channels, height * 2, width * 2), device=x_laplacian.device)
        out[:, :, ::2, ::2] = x_gaussian
        out = self.gaussian_blur_up(out)
        out = torch.add(out, x_laplacian)

        return out


class _ChannelAttentionLayer(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super(_ChannelAttentionLayer, self).__init__()
        self.channel_attention_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.channel_attention_layer(x)

        out = torch.mul(out, identity)

        return out


class _ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super(_ResidualChannelAttentionBlock, self).__init__()
        self.residual_channel_attention_block = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            _ChannelAttentionLayer(channels, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.residual_channel_attention_block(x)

        out = torch.add(out, identity)

        return out


class _ResidualGroup(nn.Module):
    def __init__(self, channels: int, reduction: int, num_rcab: int):
        super(_ResidualGroup, self).__init__()
        residual_group = []

        for _ in range(num_rcab):
            residual_group.append(_ResidualChannelAttentionBlock(channels, reduction))
        residual_group.append(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)))

        self.residual_group = nn.Sequential(*residual_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.residual_group(x)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


def realsr_rcan_x2(**kwargs: Any) -> RealSRRCAN:
    model = RealSRRCAN(upscale_factor=2, **kwargs)

    return model


def realsr_rcan_x3(**kwargs: Any) -> RealSRRCAN:
    model = RealSRRCAN(upscale_factor=3, **kwargs)

    return model


def realsr_rcan_x4(**kwargs: Any) -> RealSRRCAN:
    model = RealSRRCAN(upscale_factor=4, **kwargs)

    return model
