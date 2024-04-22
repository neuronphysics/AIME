import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from ml_collections.config_dict import config_dict

from VRNN.perceiver import perceiver

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
