from collections import OrderedDict
import math
import torch.nn as nn
import torch
from typing import Literal, Tuple
from torch import Tensor
import numpy as np
import itertools


class ModuleOutput(OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        output.last_hidden_state = self.dropout(output.last_hidden_state) + args[0]
        return output


def init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)

def _gn_groups(c, max_groups=8):
    g = math.gcd(c, max_groups)  # greatest common divisor with 8
    return g if g > 0 else 1

def _finite_stats(name, *tensors, verbose=False):
    for t in tensors:
        if t is None:
            print(f"[{name}] is None")
            continue
        if t.numel() == 0:
            print(f"[EMPTY {name}] shape={tuple(t.shape)}  (numel==0)")
            raise ValueError(f"{name} is empty (numel==0). Upstream shape logic likely wrong.")
        bad = (~torch.isfinite(t))
        if bad.any():
            idx = bad.nonzero(as_tuple=False)[:8].tolist()
            print(
                f"[NaN/Inf@{name}] shape={tuple(t.shape)} "
                f"min={float(torch.nanmin(t))} max={float(torch.nanmax(t))} "
                f"examples={idx}"
            )
            raise FloatingPointError(f"{name} produced NaN/Inf")
        else:
            if verbose:
                print(
                    f"[OK {name}] mean={float(t.mean())} "
                    f"std={float(t.std(unbiased=False))} "
                    f"min={float(t.min())} max={float(t.max())}"
                )

def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods



class BroadcastedPositionEmbedding(nn.Module):
    r"""Spatiotemporal broadcasted positional embeddings.

    Based on broadcasted position embedding algorithm in codebase:
        https://github.com/wilson1yan/VideoGPT/blob/c21cc7e2579f820cb2b90097406d72cf69a46474/videogpt/attention.py#L458

        Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \prod_{j>i}\text{dim}[j]`.

    Args:
        latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
        embedding_dim (int): The size of each embedding vector.

    Raises:
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        """
        Args:
            latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
            embedding_dim (int): The size of each embedding vector.

        Raises:
            ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``
        """
        super().__init__()
        if embedding_dim % len(latent_shape) != 0:
            raise ValueError(
                f"Embedding dim {embedding_dim} modulo len(latent_shape) {len(latent_shape)} is not zero"
            )

        self.latent_shape = latent_shape
        self.n_dim = n_dim = len(self.latent_shape)
        self.embedding_dim = embedding_dim

        self.embedding = nn.ParameterDict(
            {
                f"d_{i}": nn.Parameter(
                    torch.randn(self.latent_shape[i], embedding_dim // n_dim) * 0.01
                )
                for i in range(n_dim)
            }
        )

    @property
    def indices(self) -> Tensor:
        """Returns broadcasted indices of the data

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 3), embedding_dim=6)
            >>> pos_emb.indices
            tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        """
        device = next(self.parameters()).device
        return torch.cartesian_prod(*[torch.arange(s, device=device) for s in self.latent_shape])

    def _broadcast(self, i: int) -> Tensor:
        """Broadcasts the ``i``-th embedding matrix ``(self.latent_shape[i], self.embedding_dim // n_dim)`` along the other
        dims of ``self.latent_shape``. The embedding dim is not touched.

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 4), embedding_dim=6)
            >>> print(pos_emb.embedding["d_0"].shape)
            torch.Size([2, 3])
            >>> pos_emb.embedding["d_0"] = nn.Parameter(torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
            >>> out = pos_emb._broadcast(i=0)
            >>> print(out)
            tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]])
            >>> print(out.shape)
            (2, 4, 3)

        The input is broadcasted along the second dim ``4`` since it's the ``0``-th embedding constructed w.r.t the
        first dim ``2``.
        """
        emb = self.embedding[f"d_{i}"]
        # (1, ..., 1, self.latent_shape[i], 1, ..., embedding_dim)
        emb = emb.view(
            *itertools.repeat(1, i),
            self.latent_shape[i],
            *itertools.repeat(1, (self.n_dim - i - 1)),
            -1,
        )
        # (*self.latent_shape, embedding_dim)
        emb = emb.expand(*self.latent_shape, -1)

        return emb

    def forward(self, position_ids: Tensor) -> Tensor:
        """
        Args:
            position_ids (Tensor): batches of of 1D integer tensors indicating locations of the broadcasted
                position embeddings to be returned.

        Returns:
            A tensor with the position embeddings selected by position ids.

        Raises:
            IndexError: If any position id(s) provided is outside of the indices range.
        """
        invalid_ids = position_ids[
            torch.logical_or(position_ids >= len(self.indices), position_ids < -1)
        ]
        if len(invalid_ids):
            raise IndexError(f"Invalid position ids: {invalid_ids}")

        embeddings = []
        for i in range(self.n_dim):
            emb = self._broadcast(i)
            embeddings.append(emb)

        # concatenated embeddings: (*(shape), embedding_dim)
        embeddings = torch.cat(embeddings, dim=-1)

        # expand the permuted tensor to form a list of size `n_dim`
        # where each elm is a tensor of shape (pos_ids, batch)
        indices = [*self.indices[position_ids].permute(2, 1, 0)]
        embeddings = embeddings[indices].transpose(0, 1)  # (batch, pos_ids, emb_dim)

        return embeddings

