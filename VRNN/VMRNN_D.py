import time
import math
import copy
from functools import partial
from typing import Optional, Callable
from timm.models.swin_transformer import PatchEmbed, PatchMerging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        use_checkpoint=False,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_checkpoint = bool(use_checkpoint)
        self.ckpt_use_reentrant = False
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def _maybe_ckpt(self, fn, *args):
        if (
            self.use_checkpoint
            and self.training
            and any(torch.is_tensor(arg) and arg.requires_grad for arg in args)
        ):
            return checkpoint.checkpoint(
                fn,
                *args,
                use_reentrant=self.ckpt_use_reentrant,
            )
        return fn(*args)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L) # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        #assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv2d(x))# (b, d, h, w)
        return self.forward_core(x)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        y = self._maybe_ckpt(self._core, x)
        
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, use_checkpoint=use_checkpoint, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic VSS layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_checkpoint=use_checkpoint,
                **kwargs,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1, 2) # B H W C -> B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



class VSB(VSSBlock):
    def __init__(
        self,
        hidden_dim: int = 0,
        input_resolution: tuple = (224, 224), 
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        use_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            input_resolution=input_resolution,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            use_checkpoint=use_checkpoint,
            **kwargs
        )
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_resolution = input_resolution

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.ln_1(x)

        if hx is not None:
            hx = self.ln_1(hx)
            x = torch.cat((x, hx), dim=-1)
            x = self.linear(x)
        x = x.view(B, H, W, C) 

        x = self.drop_path(self.self_attention(x))
 
        x = x.view(B, H * W, C)
        x = shortcut + x

        return x



# cite the 'PatchExpand' code from Swin-Unet, Thanks!
# https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/
# swin_transformer_unet_skip_expand_decoder_sys.py, line 333-355
class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x




class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)

        return x


class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution,depth,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, d_state:int =16, use_checkpoint: bool = False, **kwargs):
        """
        Args:
        hidden_dim: Dimension of the hidden layer.
        input_resolution: Tuple of the input resolution.
        depth: Depth of the cell.
        drop, attn_drop, drop_path: Parameters for VSB.
        norm_layer: Normalization layer.
        d_state: State dimension for SS2D in VSB.
        """
        super(VMRNNCell, self).__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.VSBs = nn.ModuleList(
            VSB(hidden_dim=hidden_dim, input_resolution=input_resolution,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop,
                d_state=d_state, use_checkpoint=self.use_checkpoint, **kwargs)
            for i in range(depth))

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)
        else:
            hx, cx = hidden_states
        
        outputs = []
        for index, layer in enumerate(self.VSBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                x = layer(outputs[-1], None)  # Assuming VSB does not use hx for layers after the first
                outputs.append(x)
                
        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)

        cell = torch.tanh(o_t)

        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)



class VMRNNCore(nn.Module):
    def __init__(
        self,
        zdim: int,
        action_dim: int,
        height: int = 32,
        width: int = 32,
        depth: int = 4,
        d_state: int = 16,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.zdim = int(zdim)
        self.action_dim = int(action_dim)
        self.height = int(height)
        self.width = int(width)

        self.hidden_dim = self.zdim + self.action_dim

        self.action_norm = nn.LayerNorm(self.action_dim)

        self.input_norm = nn.LayerNorm(self.hidden_dim, eps=1e-5)
        self.num_tokens = self.height * self.width
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, self.zdim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.cell = VMRNNCell(
            hidden_dim=self.hidden_dim,
            input_resolution=(self.height, self.width),
            depth=depth,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            d_state=d_state,
            use_checkpoint=use_checkpoint,
        )

        # Apply this on token tensor [B, L, hidden_dim], not on [B, C, H, W].
        self.out_norm = nn.LayerNorm(self.hidden_dim, eps=1e-5)

    def init_state(self, B, device, dtype):

        h = torch.zeros(B, self.num_tokens, self.hidden_dim, device=device, dtype=dtype)
        c = torch.zeros(B, self.num_tokens, self.hidden_dim, device=device, dtype=dtype)
        return h, c

    def forward(self, z_map, a_t, state=None, mask_t=None, extra_maps=None):
        B, C, H, W = z_map.shape

        assert C == self.zdim, f"Expected zdim={self.zdim}, got C={C}"
        assert H == self.height and W == self.width
        assert extra_maps is None, "No extra maps in the no-projection version."

        if state is None:
            h, c = self.init_state(B, z_map.device, z_map.dtype)
        else:
            h, c = state

        if mask_t is not None:
            keep = mask_t.view(B, 1, 1).to(device=z_map.device, dtype=z_map.dtype)
            h0, c0 = self.init_state(B, z_map.device, z_map.dtype)
            h = h * keep + h0 * (1.0 - keep)
            c = c * keep + c0 * (1.0 - keep)

        # [B, zdim, H, W] -> [B, H*W, zdim]
        z_tok = z_map.flatten(2).transpose(1, 2).contiguous()

        # ------------------------------------------------------------
        # 2. Add simple positional embedding to latent channels only
        # ------------------------------------------------------------
        # pos_emb: [1, H*W, zdim]
        # z_tok:  [B, H*W, zdim]
        #
        # This is not RoPE. This is plain additive absolute position.
        # It works for VMRNN/SS2D because the scan block receives
        # position-aware tokens directly.
        z_tok = z_tok + self.pos_emb.to(
            device=z_tok.device,
            dtype=z_tok.dtype,
        )
        # [B, action_dim] -> [B, H*W, action_dim]
        a_t = a_t.to(device=z_map.device, dtype=z_map.dtype)
        a_t = self.action_norm(a_t)
        a_tok = a_t[:, None, :].expand(B, self.num_tokens, self.action_dim)

        # [B, H*W, zdim + action_dim]
        x_tok = torch.cat([z_tok, a_tok], dim=-1)

        x_tok = self.input_norm(x_tok)

        # VMRNNCell outputs: Ht, (Ht, Ct)
        h_out, (h_new, c_new) = self.cell(x_tok, (h, c))

        h_out = self.out_norm(h_out)

        # [B, 1024, hidden_dim] -> [B, hidden_dim, 32, 32]
        h_map = h_out.transpose(1, 2).reshape(B, self.hidden_dim, self.height, self.width).contiguous()

        return h_map, (h_new, c_new)


class DownSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, d_state=16, flag=1):
        super(DownSample, self).__init__()

        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                      dim=int(embed_dim * 2 ** i_layer))

            layer = VMRNNCell(hidden_dim=int(embed_dim * 2 ** i_layer),
                                  input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                  depth=depths_downsample[i_layer],
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[sum(depths_downsample[:i_layer]):sum(depths_downsample[:i_layer + 1])],
                                  norm_layer=norm_layer, d_state=d_state, flag=flag)

            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):

        x = self.patch_embed(x)

        hidden_states_down = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)

        return hidden_states_down, x


class UpSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_upsample, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, d_state=16, flag=0):
        super(UpSample, self).__init__()

        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.grid_size
        self.Unembed = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            resolution1 = (patches_resolution[0] // (2 ** (self.num_layers - i_layer)))
            resolution2 = (patches_resolution[1] // (2 ** (self.num_layers - i_layer)))

            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(resolution1, resolution2), dim=dimension)

            layer = VMRNNCell(hidden_dim=dimension, input_resolution=(resolution1, resolution2),
                                  depth=depths_upsample[(self.num_layers - 1 - i_layer)],drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[sum(depths_upsample[:(self.num_layers - 1 - i_layer)]):
                                                                          sum(depths_upsample[:(self.num_layers - 1 - i_layer) + 1])],
                                  norm_layer=norm_layer, d_state=d_state, flag=flag)

            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(self, x, y):
        hidden_states_up = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)

        x = torch.sigmoid(self.Unembed(x))

        return hidden_states_up, x


class VMRNN(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, depths_upsample, **kwargs):
        super(VMRNN, self).__init__()

        self.Downsample = DownSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                     embed_dim=embed_dim, depths_downsample=depths_downsample,
                                     **kwargs)

        self.Upsample = UpSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                 embed_dim=embed_dim, depths_upsample=depths_upsample,
                                 **kwargs)

    def forward(self, input, states_down, states_up,**kwargs):
        states_down, x = self.Downsample(input, states_down)
        states_up, output = self.Upsample(x, states_up)

        return output, states_down, states_up
 