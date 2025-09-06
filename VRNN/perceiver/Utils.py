import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from ml_collections.config_dict import config_dict
import cv2
from VRNN.perceiver import perceiver
import math
from typing import Literal, Sequence, cast
from torch import nn, Tensor
DEFAULT_MODEL_KWARGS = config_dict.ConfigDict({
    'HiPClassBottleneck': {
        # The size of the raw ('latent') position encodings.
        # If != the embedding size, will be projected.
        'num_position_encoding_channels': 512,
        'regroup_type': 'reshape',
        'activation_name': 'sq_relu',
        'dropout_prob': 0.3,
        'drop_path_rate': 0.0,
        'label_modalities': {}
    },
})


def generate_model(model_base_name, model_variant_name, mock_data):
    return perceiver.build_perceiver(
        input_data=mock_data,
        model_base_name=model_base_name,
        model_variant_name=model_variant_name,
        model_kwargs=DEFAULT_MODEL_KWARGS[model_base_name])


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def divide_into_grids(image_tensor, grid_size):
    batch_size, width, height, channels = image_tensor.shape
    num_horizontal_grids = width // grid_size
    num_vertical_grids = height // grid_size

    image_grid = image_tensor.reshape(batch_size, num_horizontal_grids, grid_size, num_vertical_grids,
                                      grid_size, channels)

    return image_grid


def random_mask_image_grid(image, grid_size, mask_percent, num_channel):
    # 1 - mask percent remaining
    grid_image = divide_into_grids(image, grid_size)
    batch_size, num_vertical_grids, _, num_horizontal_grids, _, _ = grid_image.shape

    # Calculate the total number of grids
    total_grids = num_vertical_grids * num_horizontal_grids

    # Determine the number of grids to mask out
    num_grids_to_mask = int(total_grids * mask_percent)

    # Randomly select grids to mask out
    grid_indices_to_mask = random.sample(range(total_grids), num_grids_to_mask)

    # Create a mask tensor to indicate which grids are masked out
    mask = torch.ones_like(grid_image)
    for idx in grid_indices_to_mask:
        # Calculate the row and column index of the grid
        row_idx = idx // num_horizontal_grids
        col_idx = idx % num_horizontal_grids

        # Set the corresponding grid to zero
        mask[:, row_idx, :, col_idx, :, :] = 0
    masked_image = grid_image * mask

    return masked_image.reshape(masked_image.shape[0], masked_image.shape[1] * masked_image.shape[2],
                                num_channel)


def random_mask_image_group(images, group_size, mask_percent, num_channel):
    # get group mask
    mask = torch.rand([images.shape[0], images.shape[1], group_size]) > mask_percent

    # repeat for channels
    mask = mask.unsqueeze(-1)
    mask = mask.expand(-1, -1, -1, num_channel)

    # repeat for groups
    mask = mask.repeat(1, 1, int(images.shape[1] / group_size), 1)
    masked_image = images * mask

    return masked_image.reshape(masked_image.shape[0], masked_image.shape[1] * masked_image.shape[2],
                                num_channel)

def random_mask_image_dot(image, grid_size, circle_mask_percent, num_channel):
    # Calculate the number of pixels to select
    num_pixels = int(np.prod(image.shape[:-1]) * circle_mask_percent)

    # Select random pixels
    indices = np.random.choice(np.prod(image.shape[:-1]), num_pixels, replace=False)
    indices = np.unravel_index(indices, image.shape[:-1])

    # Create a mask
    mask = np.ones_like(image[..., 0], dtype=np.uint8)

    # Apply circle mask around the selected pixels
    for i in range(len(indices[0])):
        center_x = indices[1][i]
        center_y = indices[2][i]
        cv2.circle(mask[indices[0][i]], (center_x, center_y), grid_size, 0, -1)

    expended_mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)
    # Apply the mask to the image
    masked_image = image * expended_mask

    return masked_image.reshape(masked_image.shape[0], masked_image.shape[1] * masked_image.shape[2],
                                num_channel)

def _apply_non_ragged(mask: Tensor, x: Tensor) -> Tensor:
    N, _, D = x.shape
    return torch.masked_select(x, mask.view(N, -1, 1)).reshape(N, -1, D)


def _apply_with_fill(mask: Tensor, x: Tensor, fill_value: float | Tensor) -> Tensor:
    N, L, _ = x.shape
    fill_value = fill_value.type_as(x) if isinstance(fill_value, Tensor) else fill_value
    mask = mask.view(N, L, 1)
    return torch.where(mask, x, fill_value)

def _apply_ragged(mask: Tensor, x: Tensor, padding_value: float | Tensor) -> Tensor:
    N, _, D = x.shape

    # Build indices where we want to put non-padding values
    unmasked_count = mask.sum(dim=-1)
    max_tokens = cast(int, unmasked_count.max())
    indices = torch.stack(
        [
            torch.arange(N, device=x.device).view(N, 1).expand(-1, max_tokens),
            torch.arange(max_tokens, device=x.device).view(1, max_tokens).expand(N, -1),
        ],
        dim=-1,
    )
    indices = indices[indices[..., -1] < unmasked_count.view(-1, 1)]

    if isinstance(padding_value, Tensor):
        o = padding_value.type_as(x).broadcast_to((N, max_tokens, D))
    else:
        o = x.new_full((N, max_tokens, D), padding_value)
    return torch.index_put(o, indices.unbind(-1), x[mask])

def mask_is_ragged(mask: Tensor) -> bool:
    r"""Checks if the mask is ragged.

    A mask is ragged if the number of unmasked tokens is not the same for all batch elements.

    Args:
        mask: Mask tensor to check

    Shapes:
        mask - :math:`(N, L)` where :math:`L` is the number of tokens
    """
    counts = mask.sum(dim=-1)
    return cast(bool, (counts != counts[0]).any())


def apply_mask(
    mask: Tensor,
    x: Tensor,
    fill_value: float | Tensor | None = None,
    padding_value: float | Tensor = 0,
) -> Tensor:
    r"""Apply the mask to tokens.

    It is expected that ``True`` indicates an unmasked token and ``False`` indicates a masked token.
    When ``fill_value=None`` and the mask is ragged, the result is padded to match the number of tokens in the
    largest batch element. Padding is done using ``padding_value`` and is applied to the end of each batch sequence.

    Args:
        mask: Mask tensor
        x: Input tensor
        fill_value: Value to fill the masked tokens with. If ``None``, the masked tokens are removed.
        padding_value: Padding value used when the mask is ragged.

    Shapes:
        mask - :math:`(N, L)` where :math:`L` is the number of tokens
        x - :math:`(N, L, D)`
        Output - :math:`(N, L', D)` where :math:`L'` is the number of output tokens

    Returns:
        Tensor with the mask applied
    """
    if x.shape[:-1] != mask.shape:
        raise ValueError(
            f"Mask and input must match in all dimensions except the last: {x.shape} != {mask.shape}"
        )  # pragma: no cover

    if fill_value is not None:
        return _apply_with_fill(mask, x, fill_value)
    elif not mask_is_ragged(mask):
        return _apply_non_ragged(mask, x)
    else:
        return _apply_ragged(mask, x, padding_value)


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
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