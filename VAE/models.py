import torch
import torch.nn as nn
from typing import Optional, List
import logging
import functools
from torch.utils.checkpoint import checkpoint

def get_improved_scheduler(optimizer, warmup_steps=10000):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return min(1.0, step / warmup_steps)
        return max(0.1, 1.0 - (step - warmup_steps) / (100000 - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LinearResidual(nn.Module):
    def __init__(self, input_feature, nonlinearity=None, norm_type='layer'):
        super(LinearResidual, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        layers = []
        layers.append(nn.Linear(input_feature, input_feature))

        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(input_feature, affine=True))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(input_feature))

        layers.append(nl)

        layers.append(nn.Linear(input_feature, input_feature))

        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(input_feature, affine=True))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(input_feature))

        layers.append(nl)

        self.fn = nn.Sequential(*layers)

    def forward(self, x):
        return self.fn(x) + x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_channels,
        *,
        num_layers=2,
        kernel_size=3,
        dilation=1,
        groups=1,
        rezero=True,
    ):
        super().__init__()
        ch = n_channels
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        layers = []
        for i in range(num_layers):
            layers.extend(
                [
                    nn.LeakyReLU(1e-2),
                    nn.Conv2d(
                        ch,
                        ch,
                        kernel_size=kernel_size,
                        padding=pad,
                        dilation=dilation,
                        groups=groups,
                    ),
                    nn.GroupNorm(1, ch),  # <--- Insert norm here
                ]
            )
        self.net = nn.Sequential(*layers)
        if rezero:
            self.gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.gate = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs) * self.gate


def log_residual_stack_structure(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,
) -> List[str]:
    logging.debug(f"Creating structure with {downsample} downsamples.")
    out = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            out.append(
                "Residual Block with "
                "{} channels and "
                "{} layers.".format(
                    channel_size_per_layer[layer], layers_per_block_per_layer[layer]
                )
            )
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    out.append(
                        "Con2d layer with "
                        "{} input channels and "
                        "{} output channels".format(
                            channel_size_per_layer[layer - 1],
                            channel_size_per_layer[layer],
                        )
                    )
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

        # after the residual block, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                out.append("Avg Pooling layer.")
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                out.append("Interpolation layer.")

    return out


def build_residual_stack(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,
) -> List[nn.Module]:
    logging.debug(
        "\n".join(
            log_residual_stack_structure(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=encoder,
            )
        )
    )
    layers = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            # add a residual block with the required number of channels and layers
            layers.append(
                ResidualBlock(
                    channel_size_per_layer[layer],
                    num_layers=layers_per_block_per_layer[layer],
                )
            )
            layers.append(nn.GroupNorm(1, channel_size_per_layer[layer]))
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

                    in_channels = channel_size_per_layer[layer - 1]
                    out_channels = channel_size_per_layer[layer]
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # after the residual blocks, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                layers.append(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                layers.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )

    return layers

class attentionBlock(nn.Module):
    def __init__(self, n_emb, n_heads=4):
        super().__init__()
        self.flatten = nn.Flatten(2)
        #self.n_input = n_input
        self.n_emb = n_emb
        self.norm = nn.GroupNorm(4, n_emb)
        self.attention = nn.MultiheadAttention(n_emb, n_heads, bias=True,  batch_first=True)

    def forward(self, x):
        batch_size, n_channels, h, w = x.size()
        residue = x
        x = self.norm(x)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(batch_size, n_channels, h, w)
        
        return x + residue

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class ImageDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=16, n_layers=5, use_actnorm=False, device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ImageDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            if n==(n_layers-1):
               sequence += [attentionBlock(ndf* nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        self.to(device)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class LatentDiscriminator(nn.Module):
    # define the descriminator/critic
    def __init__(self, input_dims, num_layers=4, norm_type='layer', activation= nn.GELU(), device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(LatentDiscriminator, self).__init__()
        self.norm_type = norm_type
        self.activation = activation
        layers = []
        layers.append(nn.Linear(input_dims, input_dims * 2, bias=False))

        if self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(input_dims * 2))
        elif self.norm_type == 'layer':
            layers.append(nn.LayerNorm(input_dims * 2))
        # Activation Function
        layers.append(self.activation)
        size = input_dims * 2
        # Fully Connected Block
        for i in range(num_layers - 2):
            # residual feedforward Layer
            layers.append(nn.Linear(size, size // 2))

            if self.norm_type == 'batch':
                layers.append(nn.BatchNorm1d(size // 2))
            elif self.norm_type == 'layer':
                layers.append(nn.LayerNorm(size // 2))

            layers.append(self.activation)
            if (i == (num_layers // 2 - 1)):
                # add a residual block
                layers.append(LinearResidual(size // 2))
            size = size // 2
        layers.append(nn.Linear(size, size * 2, bias=False))

        if self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(size * 2))
        elif self.norm_type == 'layer':
            layers.append(nn.LayerNorm(size * 2))

        # Activation Function
        layers.append(self.activation)
        # add anther residual block
        layers.append(LinearResidual(size * 2))
        layers.append(nn.Linear(size * 2, 1))
        self.model = nn.Sequential(*layers)
        self.device = device
        self.to(device=self.device)

    def forward(self, x):
        return self.model(x)


class VAEEncoder(torch.nn.Module):
    def __init__(
        self,
        channel_size_per_layer: List[int],
        layers_per_block_per_layer: List[int],
        latent_size: int,
        width: int,
        height: int,
        num_layers_per_resolution,
        mlp_hidden_size: int = 512,
        channel_size: int = 64,
        input_channels: int = 3,
        downsample: int = 4,
    ):
        super().__init__()
        self.latent_size = latent_size

        # compute final width and height of feature maps
        inner_width = width // (2**downsample)
        inner_height = height // (2**downsample)

        # conv layers
        layers = [
            nn.Conv2d(input_channels, channel_size, 5, padding=2, stride=2),
            nn.GELU(),
        ]
        
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample - 1,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=True,
            )
        )

        mlp_input_size = channel_size_per_layer[-1] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers.extend(
            [
                nn.Flatten(),
                nn.GELU(),
                nn.Linear(mlp_input_size, mlp_hidden_size),
                nn.GELU(),
                nn.LayerNorm(mlp_hidden_size),
            ]
        )

        self.net = nn.Sequential(*layers)
        self.mu  = nn.Linear(mlp_hidden_size, latent_size)
        self.logvar = nn.Linear(mlp_hidden_size, latent_size)
        
    def gradient_checkpointing_enable(self):
        for module in self.net:
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor) -> dict:
        self.hlayer = self.net(x)
        mean = self.mu(self.hlayer)
        logvar = self.logvar(self.hlayer)
        sigma = (logvar * 0.5).exp()
        latent_normal = torch.distributions.Normal(mean, sigma)
        z = latent_normal.rsample()  # [Batch size, Latent size]
        return z, mean, logvar



class VAEDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size: int,
        width: int,
        height: int,
        channel_size_per_layer: List[int] = (256, 256, 256, 256, 128, 128, 64, 64),
        layers_per_block_per_layer: List[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        num_layers_per_resolution: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
        downsample: Optional[int] = 4,
        mlp_hidden_size: Optional[int] = 512,
    ):
        super().__init__()
        # Add memory-efficient settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # compute final width and height of feature maps
        inner_width = width // (2**downsample)
        inner_height = height // (2**downsample)

        mlp_input_size = channel_size_per_layer[0] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers = []
        layers.extend(
            [
                nn.Linear(latent_size, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, mlp_input_size),
                nn.Unflatten(
                    1,
                    unflattened_size=(
                        channel_size_per_layer[0],
                        inner_height,
                        inner_width,
                    ),
                ),
                # B, 64*4, 4, 4
            ]
        )

        # conv layers
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=False,
            )
        )
        layers.append(nn.BatchNorm2d(channel_size_per_layer[-1]))
        layers.append(nn.GELU())
        final_conv = nn.Conv2d(channel_size_per_layer[-1], input_channels, 5, padding=2)
        
        layers.extend([
                        final_conv,
                        nn.Tanh()  # range of image pixel values between [-1,1]
                      ])
        
        self.net = nn.Sequential(*layers)
        
    def gradient_checkpointing_enable(self):
        for module in self.net:
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
                
    # @torch.compile  # Add torch.compile for faster inference
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

