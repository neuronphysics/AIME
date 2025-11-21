# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#https://github.com/sgl-project/sglang
## ==============================================================================
from typing import Any, Tuple, Union
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as ckpt

def base_group_norm(x, norm_layer, act_silu=False, channel_last=False) -> torch.Tensor:
    if hasattr(base_group_norm, "spatial") and base_group_norm.spatial:
        assert channel_last
        x_shape = x.shape
        x = x.flatten(0, 1)
        if channel_last:
            # Permute to NCHW format
            x = x.permute(0, 3, 1, 2)

        out = F.group_norm(
            x.contiguous(),
            norm_layer.num_groups,
            norm_layer.weight,
            norm_layer.bias,
            norm_layer.eps,
        )
        if act_silu:
            out = F.silu(out)

        if channel_last:
            # Permute back to NHWC format
            out = out.permute(0, 2, 3, 1)

        out = out.view(x_shape)
    else:
        if channel_last:
            # Permute to NCHW format
            x = x.permute(0, 3, 1, 2)
        out = F.group_norm(
            x.contiguous(),
            norm_layer.num_groups,
            norm_layer.weight,
            norm_layer.bias,
            norm_layer.eps,
        )
        if act_silu:
            out = F.silu(out)
        if channel_last:
            # Permute back to NHWC format
            out = out.permute(0, 2, 3, 1)
    return out


def base_conv2d(x, conv_layer, channel_last=False, residual=None) -> torch.Tensor:
    if channel_last:
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
    out = F.conv2d(
        x,
        conv_layer.weight,
        conv_layer.bias,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
    )
    if residual is not None:
        if channel_last:
            residual = residual.permute(0, 3, 1, 2)  # NHWC to NCHW
        out += residual
    if channel_last:
        out = out.permute(0, 2, 3, 1)  # NCHW to NHWC
    return out


def base_conv3d(
    x, conv_layer, channel_last=False, residual=None, only_return_output=False
) -> torch.Tensor:
    if only_return_output:
        size = cal_outsize(
            x.shape, conv_layer.weight.shape, conv_layer.stride, conv_layer.padding
        )
        return torch.empty(size, device=x.device, dtype=x.dtype)
    if channel_last:
        x = x.permute(0, 4, 1, 2, 3)  # NDHWC to NCDHW
    out = F.conv3d(
        x,
        conv_layer.weight,
        conv_layer.bias,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
    )
    if residual is not None:
        if channel_last:
            residual = residual.permute(0, 4, 1, 2, 3)  # NDHWC to NCDHW
        out += residual
    if channel_last:
        out = out.permute(0, 2, 3, 4, 1)  # NCDHW to NDHWC
    return out


def cal_outsize(input_sizes, kernel_sizes, stride, padding) -> list:
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = 1, 1, 1

    in_d = input_sizes[1]
    in_h = input_sizes[2]
    in_w = input_sizes[3]

    kernel_d = kernel_sizes[2]
    kernel_h = kernel_sizes[3]
    kernel_w = kernel_sizes[4]
    out_channels = kernel_sizes[0]

    out_d = calc_out_(in_d, padding_d, dilation_d, kernel_d, stride_d)
    out_h = calc_out_(in_h, padding_h, dilation_h, kernel_h, stride_h)
    out_w = calc_out_(in_w, padding_w, dilation_w, kernel_w, stride_w)
    size = [input_sizes[0], out_d, out_h, out_w, out_channels]
    return size


def calc_out_(
    in_size: int, padding: int, dilation: int, kernel: int, stride: int
) -> int:
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def base_conv3d_channel_last(x, conv_layer, residual=None) -> torch.Tensor:
    in_numel = x.numel()
    out_numel = int(x.numel() * conv_layer.out_channels / conv_layer.in_channels)
    if (in_numel >= 2**30) or (out_numel >= 2**30):
        assert conv_layer.stride[0] == 1, "time split asks time stride = 1"

        B, T, H, W, C = x.shape
        K = conv_layer.kernel_size[0]

        chunks = 4
        chunk_size = T // chunks

        if residual is None:
            out_nhwc = base_conv3d(
                x,
                conv_layer,
                channel_last=True,
                residual=residual,
                only_return_output=True,
            )
        else:
            out_nhwc = residual

        assert B == 1
        for i in range(chunks):
            if i == chunks - 1:
                xi = x[:1, chunk_size * i :]
                out_nhwci = out_nhwc[:1, chunk_size * i :]
            else:
                xi = x[:1, chunk_size * i : chunk_size * (i + 1) + K - 1]
                out_nhwci = out_nhwc[:1, chunk_size * i : chunk_size * (i + 1)]
            if residual is not None:
                if i == chunks - 1:
                    ri = residual[:1, chunk_size * i :]
                else:
                    ri = residual[:1, chunk_size * i : chunk_size * (i + 1)]
            else:
                ri = None
            out_nhwci.copy_(base_conv3d(xi, conv_layer, channel_last=True, residual=ri))
    else:
        out_nhwc = base_conv3d(x, conv_layer, channel_last=True, residual=residual)
    return out_nhwc


class Upsample2D(nn.Module):

    def __init__(
        self, channels, use_conv=False, use_conv_transpose=False, out_channels=None
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        else:
            assert "Not Supported"
            self.conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)

    def forward(self, x, output_size=None) -> torch.Tensor:
        assert x.shape[-1] == self.channels

        if self.use_conv_transpose:
            return self.conv(x)

        if output_size is None:
            x = (
                F.interpolate(
                    x.permute(0, 3, 1, 2).to(memory_format=torch.channels_last),
                    scale_factor=2.0,
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        else:
            x = (
                F.interpolate(
                    x.permute(0, 3, 1, 2).to(memory_format=torch.channels_last),
                    size=output_size,
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        # x = self.conv(x)
        x = base_conv2d(x, self.conv, channel_last=True)
        return x


class Downsample2D(nn.Module):

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x) -> torch.Tensor:
        assert x.shape[-1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 0, 0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)

        assert x.shape[-1] == self.channels
        # x = self.conv(x)
        x = base_conv2d(x, self.conv, channel_last=True)
        return x


class CausalConv(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size, **kwargs) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (
                kernel_size if isinstance(kernel_size, tuple) else ((kernel_size,) * 3)
            )
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.dilation = kwargs.pop("dilation", 1)
        self.stride = kwargs.pop("stride", 1)
        if isinstance(self.stride, int):
            self.stride = (self.stride, 1, 1)
        time_pad = self.dilation * (time_kernel_size - 1) + max((1 - self.stride[0]), 0)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )
        self.time_uncausal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            0,
            0,
        )

        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            **kwargs,
        )
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.is_first_run = True

    def forward(self, x, is_init=True, residual=None) -> torch.Tensor:
        x = nn.functional.pad(
            x, self.time_causal_padding if is_init else self.time_uncausal_padding
        )
        x = self.conv(x)
        if residual is not None:
            x.add_(residual)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**3 % in_channels == 0
        self.repeats = out_channels * factor**3 // in_channels

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor,
            self.factor,
            self.factor,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor,
            x.size(4) * self.factor,
            x.size(6) * self.factor,
        )
        x = x[:, :, self.factor - 1 :, :, :]
        return x


class ConvPixelShuffleUpSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        self.conv = CausalConv(
            in_channels, out_channels * out_ratio, kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = self.conv(x, is_init)
        x = self.pixel_shuffle_3d(x, self.factor)
        return x

    @staticmethod
    def pixel_shuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.size()
        new_channels = channels // (factor**3)
        new_depth = depth * factor
        new_height = height * factor
        new_width = width * factor

        x = x.view(
            batch_size, new_channels, factor, factor, factor, depth, height, width
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, new_channels, new_depth, new_height, new_width)
        x = x[:, :, factor - 1 :, :, :]
        return x


class ConvPixelUnshuffleDownSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        assert out_channels % out_ratio == 0
        self.conv = CausalConv(
            in_channels, out_channels // out_ratio, kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = self.conv(x, is_init)
        x = self.pixel_unshuffle_3d(x, self.factor)
        return x

    @staticmethod
    def pixel_unshuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
        pad = (0, 0, 0, 0, factor - 1, 0)  # (left, right, top, bottom, front, back)
        x = F.pad(x, pad)
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // factor, factor, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * factor**3, D // factor, H // factor, W // factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**3 % out_channels == 0
        self.group_size = in_channels * factor**3 // out_channels

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        pad = (
            0,
            0,
            0,
            0,
            self.factor - 1,
            0,
        )  # (left, right, top, bottom, front, back)
        x = F.pad(x, pad)
        B, C, D, H, W = x.shape
        x = x.view(
            B,
            C,
            D // self.factor,
            self.factor,
            H // self.factor,
            self.factor,
            W // self.factor,
            self.factor,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B, C * self.factor**3, D // self.factor, H // self.factor, W // self.factor
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            D // self.factor,
            H // self.factor,
            W // self.factor,
        )
        x = x.mean(dim=2)
        return x


def base_group_norm_with_zero_pad(
    x, norm_layer, act_silu=True, pad_size=2
) -> torch.Tensor:
    out_shape = list(x.shape)
    out_shape[1] += pad_size
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out[:, pad_size:] = base_group_norm(
        x, norm_layer, act_silu=act_silu, channel_last=True
    )
    out[:, :pad_size] = 0
    return out


class CausalConvChannelLast(CausalConv):
    time_causal_padding: tuple[Any, ...]
    time_uncausal_padding: tuple[Any, ...]

    def __init__(self, chan_in, chan_out, kernel_size, **kwargs) -> None:
        super().__init__(chan_in, chan_out, kernel_size, **kwargs)

        self.time_causal_padding = (0, 0) + self.time_causal_padding
        self.time_uncausal_padding = (0, 0) + self.time_uncausal_padding

    def forward(self, x, is_init=True, residual=None) -> torch.Tensor:
        if self.is_first_run:
            self.is_first_run = False
            # self.conv.weight = nn.Parameter(self.conv.weight.permute(0,2,3,4,1).contiguous())

        x = nn.functional.pad(
            x, self.time_causal_padding if is_init else self.time_uncausal_padding
        )

        x = base_conv3d_channel_last(x, self.conv, residual=residual)
        return x


class CausalConvAfterNorm(CausalConv):

    def __init__(self, chan_in, chan_out, kernel_size, **kwargs) -> None:
        super().__init__(chan_in, chan_out, kernel_size, **kwargs)

        if self.time_causal_padding == (1, 1, 1, 1, 2, 0):
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=(0, 1, 1),
                **kwargs,
            )
        else:
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                **kwargs,
            )
        self.is_first_run = True

    def forward(self, x, is_init=True, residual=None) -> torch.Tensor:
        if self.is_first_run:
            self.is_first_run = False

        if self.time_causal_padding == (1, 1, 1, 1, 2, 0):
            pass
        else:
            x = nn.functional.pad(x, self.time_causal_padding).contiguous()

        x = base_conv3d_channel_last(x, self.conv, residual=residual)
        return x


class Resnet3DBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels=None,
        temb_channels=512,
        conv_shortcut=False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = CausalConvAfterNorm(in_channels, out_channels, kernel_size=3)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = CausalConvAfterNorm(out_channels, out_channels, kernel_size=3)

        assert conv_shortcut is False
        self.use_conv_shortcut = conv_shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConvAfterNorm(
                    in_channels, out_channels, kernel_size=3
                )
            else:
                self.nin_shortcut = CausalConvAfterNorm(
                    in_channels, out_channels, kernel_size=1
                )
        #self.pe = ProjectExciteLayer(self.out_channels, reduction_ratio=2)

    def forward(self, x, temb=None, is_init=True) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        h = base_group_norm_with_zero_pad(x, self.norm1, act_silu=True, pad_size=2)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nn.functional.silu(temb))[:, :, None, None]

        x = self.nin_shortcut(x) if self.in_channels != self.out_channels else x

        h = base_group_norm_with_zero_pad(h, self.norm2, act_silu=True, pad_size=2)
        h = nonlinearity(h)
        x = self.conv2(h, residual=x)

        x = x.permute(0, 4, 1, 2, 3)
        #x = self.pe(x)
        
        return x


class Downsample3D(nn.Module):

    def __init__(self, in_channels, with_conv, stride) -> None:
        super().__init__()

        self.with_conv = with_conv
        if with_conv:
            self.conv = CausalConv(in_channels,
                                   in_channels,
                                   kernel_size=3,
                                   stride=stride)

    def forward(self, x, is_init=True) -> torch.Tensor:
        if self.with_conv:
            x = self.conv(x, is_init)
        else:
            x = nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class SpatialConvPixelUnshuffleDownSampleLayer3D(nn.Module):
    """
    Spatial-only 3D pixel-unshuffle downsampling.

    Input:  [B, C_in, T, H, W]
    Output: [B, C_out, T, H/2, W/2]

    Implemented as:
      - causal 3D conv:   C_in -> C_out / 4  (no stride)
      - 2D pixel-unshuffle on (H, W) per frame (factor=2)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        factor: int = 2,
    ) -> None:
        super().__init__()
        self.factor = factor
        out_ratio = factor ** 2
        assert (
            out_channels % out_ratio == 0
        ), f"out_channels ({out_channels}) must be divisible by factor^2 ({out_ratio})"

        self.conv = CausalConv(
            in_channels,
            out_channels // out_ratio,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor, is_init: bool = True) -> torch.Tensor:
        # x: [B, C_in, T, H, W]
        x = self.conv(x, is_init=is_init)        # [B, C_mid, T, H, W]
        B, C, T, H, W = x.shape
        r = self.factor
        assert H % r == 0 and W % r == 0, (
            f"H and W must be divisible by factor={r}, got H={H}, W={W}"
        )

        # pixel-unshuffle on spatial dims only
        x = rearrange(
            x,
            "b c t (h r1) (w r2) -> b (c r1 r2) t h w",
            r1=r,
            r2=r,
        )
        return x   # [B, C_out, T, H/2, W/2]


class SpatialConvPixelShuffleUpSampleLayer3D(nn.Module):
    """
    Spatial-only 3D pixel-shuffle upsampling.

    Input:  [B, C_in, T, H, W]
    Output: [B, C_out, T, H*2, W*2]

    Implemented as:
      - causal 3D conv:   C_in -> C_out * 4  (no stride)
      - 2D pixel-shuffle on (H, W) per frame (factor=2)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        factor: int = 2,
    ) -> None:
        super().__init__()
        self.factor = factor
        out_ratio = factor ** 2

        self.conv = CausalConv(
            in_channels,
            out_channels * out_ratio,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor, is_init: bool = True) -> torch.Tensor:
        # x: [B, C_in, T, H, W]
        x = self.conv(x, is_init=is_init)        # [B, C_out * 4, T, H, W]
        r = self.factor

        # pixel-shuffle on spatial dims only
        x = rearrange(
            x,
            "b (c r1 r2) t h w -> b c t (h r1) (w r2)",
            r1=r,
            r2=r,
        )
        return x   # [B, C_out, T, H*2, W*2]

class VideoEncoder(nn.Module):
    """
    x: [B, C, T, H, W]
    -> [B, T, z_channels, H', W']

    Design:
      - A stack of *spatial* sub-pixel downsampling layers (T stays the same)
      - Final CausalConv to z_channels, then [B, C, T, H, W] -> [B, T, C, H, W]
    """

    def __init__(
        self,
        ch: int,
        ch_mult,
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
        double_z: bool = False,
        resamp_with_conv: bool = True,
        kernel_size: int = 3,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert not double_z, "VideoEncoder only supports double_z=False"

        self.ch = ch
        self.z_channels = z_channels
        self.num_res_blocks = num_res_blocks
        self.resamp_with_conv = resamp_with_conv
        self.use_checkpoint = use_checkpoint

        k3d = (kernel_size, kernel_size, kernel_size)

        # 1) Initial 3D conv: RGB -> hidden
        self.conv_in = CausalConv(
            in_channels,
            ch,
            kernel_size=k3d,
        )

        # 2) ResNet blocks at input resolution
        self.res_in = nn.ModuleList(
            [Resnet3DBlock(ch, ch, temb_channels=0) for _ in range(num_res_blocks)]
        )

        # 3) Spatial downsampling tower
        #    num_spatial_down = len(ch_mult) - 1 (same logic as before)
        num_spatial_down = max(0, len(ch_mult) - 1)

        if resamp_with_conv:
            # Use spatial sub-pixel downsampling with a factor derived from ch_mult.
            # For each spatial downsampling stage i, reduce the spatial
            # dimensions by factor = ch_mult[i+1] // ch_mult[i].  This ensures
            self.down_blocks = nn.ModuleList()
            for i in range(num_spatial_down):
                # When they pass ch_mult=(1,2,4), this yields two stages each with factor=2.
                factor = ch_mult[i + 1] // ch_mult[i]
                self.down_blocks.append(
                    SpatialConvPixelUnshuffleDownSampleLayer3D(
                        in_channels=ch,
                        out_channels=ch,
                        kernel_size=k3d,
                        factor=factor,
                    )
                )
        else:
            # fallback to old avg-pool based path
            self.down_blocks = nn.ModuleList(
                [
                    Downsample3D(
                        ch,
                        with_conv=False,
                        num_res_blocks=num_res_blocks,
                        stride=(1, 2, 2),
                    )
                    
                    for _ in range(num_spatial_down)
                ]
            )
        self.res_mid = nn.ModuleList(
            [Resnet3DBlock(ch, ch, temb_channels=0) for _ in range(num_res_blocks)]
        )
        self.norm_out = Normalize(ch)
        # 4) Final projection to z_channels
        self.conv_out = CausalConv(
            ch,
            z_channels,
            kernel_size=k3d,
        )
    def maybe_checkpoint(self, fn, *x, use_reentrant: bool = False):
        """
        Wrap a single-tensor -> tensor function with torch.utils.checkpoint.

        fn: callable(tensor) -> tensor
        x:  input tensor
        """
        if self.training and self.use_checkpoint:
            return ckpt(fn, *x, use_reentrant=use_reentrant)
        else:
            return fn(*x)

    def forward(
        self,
        x: torch.Tensor,
        is_init: bool = True,
    ) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        returns: [B, T, z_channels, H', W']
        """
        # 1) Initial conv + ResNet stack
        h = self.conv_in(x, is_init=is_init)   # [B, ch, T, H, W]
        for block in self.res_in:
            def fn(h_in, block=block, is_init=is_init):
                return block(h_in, temb=None, is_init=is_init)
            h = self.maybe_checkpoint(fn, h)

        # 2) Spatial downsampling tower (T unchanged)
        for down in self.down_blocks:
            def fn(h_in, down=down, is_init=is_init):
                return down(h_in, is_init=is_init)
            h = self.maybe_checkpoint(fn, h)

        for block in self.res_mid:
            def fn(h_in, block=block, is_init=is_init):
                return block(h_in, temb=None, is_init=is_init)
            h = self.maybe_checkpoint(fn, h)


        h = self.norm_out(h)
        h = nonlinearity(h)
        # 3) Final projection to z_channels
        def cnv_out(h_in, is_init=is_init):
            return self.conv_out(h_in, is_init=is_init)
        h = self.maybe_checkpoint(cnv_out, h)  # [B, z_channels, T, H', W']

        # 4) DVAE expects [B, T, C, H, W]
        h = rearrange(h, "b c t h w -> b t c h w")
        return h

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels: int):
    # Guard against small channel counts that don't divide 32 nicely
    num_groups = min(32, in_channels)
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Res3DBlockUpsample(nn.Module):
    def __init__(self,
        input_filters,
        num_filters,
        down_sampling_stride,
        down_sampling=False
    ):
        super().__init__()
        self.input_filters = input_filters
        self.num_filters = num_filters
        self.act_ = nn.SiLU(inplace=True)

        self.conv1 = CausalConvChannelLast(num_filters, num_filters, kernel_size=[3, 3, 3])
        self.norm1 = nn.GroupNorm(32, num_filters)

        self.conv2 = CausalConvChannelLast(num_filters, num_filters, kernel_size=[3, 3, 3])
        self.norm2 = nn.GroupNorm(32, num_filters)

        self.down_sampling = down_sampling
        if down_sampling:
            self.down_sampling_stride = down_sampling_stride
        else:
            self.down_sampling_stride = [1, 1, 1]


        if num_filters != input_filters or down_sampling:
            self.conv3 = CausalConvChannelLast(input_filters, num_filters, kernel_size=[1, 1, 1], stride=self.down_sampling_stride)
            self.norm3 = nn.GroupNorm(32, num_filters)

    def forward(self, x, is_init=False):
        x = x.permute(0,2,3,4,1).contiguous()   # [B, T, H, W, C]
        residual = x

        h = self.conv1(x, is_init)
        h = base_group_norm(h, self.norm1, act_silu=True, channel_last=True)

        h = self.conv2(h, is_init)
        h = base_group_norm(h, self.norm2, act_silu=False, channel_last=True)

        if self.down_sampling or self.num_filters != self.input_filters:
            x = self.conv3(x, is_init)
            x = base_group_norm(x, self.norm3, act_silu=False, channel_last=True)

        h.add_(x)
        h = self.act_(h)
        if residual is not None:
            h.add_(residual)

        h = h.permute(0,4,1,2,3)
        return h

class Upsample3D(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv3d = Res3DBlockUpsample(
            input_filters=in_channels,
            num_filters=in_channels,
            down_sampling_stride=(1, 1, 1),
            down_sampling=False,
        )

    def forward(self, x, is_init=True, is_split=True):
        b, c, t, h, w = x.shape

        if is_split:
            split_size = c // 8
            x_slices = torch.split(x, split_size, dim=1)
            x = [nn.functional.interpolate(x, scale_factor=self.scale_factor) for x in x_slices]
            x = torch.cat(x, dim=1)
        else:
            x = nn.functional.interpolate(x, scale_factor=self.scale_factor)

        x = self.conv3d(x, is_init)
        return x

class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


############
class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class VideoDecoder(nn.Module):
    """
    3D decoder:
      Input:  [B, T, z_channels, H', W']
      Output: [B, C, T, H, W]
    Design:
      Encoder:
        x: [B, C, T, H, W]
          -> conv_in: C -> ch
          -> res_in: Resnet3DBlock(ch, ch) * num_res_blocks
          -> (spatial) down_blocks
          -> res_mid: Resnet3DBlock(ch, ch) * num_res_blocks
          -> conv_out: ch -> z_channels
          -> rearrange to [B, T, z_channels, H', W']

      Decoder (inverse):
        z: [B, T, z_channels, H', W']
          -> conv_in: z_channels -> ch
          -> res_mid at coarsest resolution
          -> spatial upsampling tower back to input res
          -> res_out at full resolution
          -> conv_out: ch -> out_channels
          -> [B, C, T, H, W]
    """

    def __init__(
        self,
        ch: int,
        z_channels: int,
        out_channels: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        kernel_size: int = 3,
        resamp_with_conv: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.ch = ch
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(self.ch_mult)
        self.resamp_with_conv = resamp_with_conv
        self.use_checkpoint = use_checkpoint

        k3d = (kernel_size, kernel_size, kernel_size)
        # Project encoder's latent z_channels back to shared hidden width ch
        

        self.conv_in = CausalConv(
            z_channels,
            ch,
            kernel_size=k3d,
        )

        # ResNet stack at coarsest resolution (mirrors VideoEncoder.res_mid)
        self.res_mid = nn.ModuleList(
            [Resnet3DBlock(ch, ch, temb_channels=0) for _ in range(num_res_blocks)]
        )

        # Number of spatial upsampling stages = len(ch_mult) - 1
        num_spatial_up = max(0, self.num_resolutions - 1)

        self.up_blocks = nn.ModuleList()
        if self.resamp_with_conv:
            # Spatial sub-pixel upsampling, T unchanged.
            # Encoder uses SpatialConvPixelUnshuffleDownSampleLayer3D with
            # factor = ch_mult[i+1] // ch_mult[i].
            # Here we mirror that via SpatialConvPixelShuffleUpSampleLayer3D
            # with the same factors, in reverse order.
            for i in reversed(range(num_spatial_up)):
                factor = self.ch_mult[i + 1] // self.ch_mult[i]
                self.up_blocks.append(
                    SpatialConvPixelShuffleUpSampleLayer3D(
                        in_channels=ch,
                        out_channels=ch,  # keep channels = ch at all resolutions
                        kernel_size=k3d,
                        factor=factor,
                    )
                )
        else:
            # Fallback: 3D upsampling (time + space), matching encoder Downsample3D
            # which uses avg_pool3d(kernel_size=2, stride=2).
            for i in range(num_spatial_up):
                self.up_blocks.append(
                    Upsample3D(
                        in_channels=ch,
                        scale_factor=2,
                    )
                )

        # ResNet stack at finest resolution (mirrors VideoEncoder.res_in)
        self.res_out = nn.ModuleList(
            [Resnet3DBlock(ch, ch, temb_channels=0) for _ in range(num_res_blocks)]
        )
        
        self.norm_out = Normalize(ch)

        # Final projection to output channels (e.g., RGB)
        self.conv_out = CausalConv(
            ch,
            out_channels,
            kernel_size=k3d,
        )
    def maybe_checkpoint(self, fn, *x, use_reentrant: bool = False):
        """
        Wrap a single-tensor -> tensor function with torch.utils.checkpoint.

        fn: callable(tensor) -> tensor
        x:  input tensor
        """
        if self.training and self.use_checkpoint:
            return ckpt(fn, *x, use_reentrant=use_reentrant)
        else:
            return fn(*x)


    def forward(self, z: torch.Tensor, is_init: bool = True) -> torch.Tensor:
        """
        z: [B, T, z_channels, H', W']
        returns: [B, C, T, H, W]
        """
        assert z.ndim == 5, f"VideoDecoder expects [B,T,C,H,W] latent, got {z.shape}"
        B, T, C, Hc, Wc = z.shape
        assert C == self.z_channels, (
            f"z_channels mismatch: got {C}, expected {self.z_channels}"
        )

        # [B, z_channels, T, H', W']
        h = rearrange(z, "b t c h w -> b c t h w")
        
        # z_channels -> ch
        def cnv_in_fn(h_in, is_init=is_init):
            return self.conv_in(h_in, is_init=is_init)
        h = self.maybe_checkpoint(cnv_in_fn, h)

        # Coarsest-resolution ResNet blocks
        for block in self.res_mid:
            def fn(h_in, block=block, is_init=is_init):
                return block(h_in, temb=None, is_init=is_init)
            h = self.maybe_checkpoint(fn, h)

        # Spatial upsampling tower
        for _, up in enumerate(self.up_blocks):
            def fn(h_in, up=up, is_init=is_init):
                return up(h_in, is_init=is_init)
            h = self.maybe_checkpoint(fn, h)

        # Finest-resolution ResNet blocks
        for block in self.res_out:
            def fn(h_in, block=block, is_init=is_init):
                return block(h_in, temb=None, is_init=is_init)
            h = self.maybe_checkpoint(fn, h)

        h = self.norm_out(h)
        h = nonlinearity(h)

        # [B, C, T, H, W]
        h = self.conv_out(h, is_init=is_init)
        return h
