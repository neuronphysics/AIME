import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple
import logging
import functools
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
class AddEpsilon(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return x + self.eps
 
@torch.jit.script
def check_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate tensor values for debugging"""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

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


class AttentionPosterior(nn.Module):
    """
    Improved posterior with explicit spatial processing
    """
    
    def __init__(
        self,
        image_size: int,
        attention_resolution: int,
        hidden_dim: int,
        context_dim: int, 
        input_channels: int = 3
    ):
        super().__init__()
        self.image_size = image_size
        self.attention_resolution = attention_resolution
        
        # Bottom-up saliency extraction
        self.saliency_net = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1)
        )
        
        # Top-down modulation
        self.top_down_projection = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )
        
        # Spatial attention computation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # Combine bottom-up and top-down
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1)  # Output attention logits
        )
        self.attention_pool = nn.Sequential(
            nn.Conv2d(64, hidden_dim//2, 1),  # Reduce channels
            nn.GroupNorm(8, hidden_dim//2),
            nn.SiLU()
        )
    
    def forward(
        self,
        observation: torch.Tensor,
        hidden_state: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        batch_size = observation.shape[0]
        
        # Bottom-up saliency
        saliency_features = self.saliency_net(observation)
        
        # Top-down modulation
        top_down = torch.cat([hidden_state, context], dim=-1)
        top_down_features = self.top_down_projection(top_down)
        top_down_spatial = top_down_features.view(batch_size, -1, 1, 1)
        top_down_spatial = top_down_spatial.expand(
            -1, -1, 
            saliency_features.shape[2], 
            saliency_features.shape[3]
        )
        
        # Combine features
        combined = torch.cat([saliency_features, top_down_spatial], dim=1)
        
        # Compute attention
        H = W = self.attention_resolution
        attention_logits = self.attention_conv(combined).squeeze(1)
        attention_probs = F.softmax(
            attention_logits.view(batch_size, -1), 
            dim=-1
        ).view(batch_size, H, W)
        
        return attention_probs

    def attention_weighted_features(
        self,
        features: torch.Tensor,  # [B, C, H, W]
        attention_probs: torch.Tensor  # [B, H, W]
    ) -> torch.Tensor:
        """
        Extract attention-weighted features for downstream processing
        """
        batch_size = features.shape[0]
        
        # Ensure attention and features have same spatial dimensions
        if attention_probs.shape[-2:] != features.shape[-2:]:
            attention_probs = F.interpolate(
                attention_probs.unsqueeze(1),
                size=features.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Apply attention weighting
        attention_expanded = attention_probs.unsqueeze(1)  # [B, 1, H, W]
        weighted_features = features * attention_expanded
        
        # Spatial pooling guided by attention
        pooled_features = self.attention_pool(weighted_features)
        
        # Weighted average pooling
        spatial_sum = (pooled_features * attention_expanded).sum(dim=[2, 3])
        normalization = attention_expanded.sum(dim=[2, 3]) + 1e-8
        
        return spatial_sum / normalization  # [B, hidden_dim//2]
    
    def visualize_attention(
        self,
        attention_map: torch.Tensor,  # [B, H, W]
        original_image: torch.Tensor,  # [B, 3, 84, 84]
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Create visualization by overlaying attention on original image
        """
        batch_size = attention_map.shape[0]
        
        # Upsample attention to image size
        attention_upsampled = F.interpolate(
            attention_map.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, 84, 84]
        
        # Normalize attention for visualization
        att_min = attention_upsampled.view(batch_size, -1).min(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        att_max = attention_upsampled.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        attention_normalized = (attention_upsampled - att_min) / (att_max - att_min + 1e-8)
        
        # Create heatmap (red channel intensified by attention)
        heatmap = torch.zeros_like(original_image)
        heatmap[:, 0, :, :] = attention_normalized.squeeze(1)  # Red channel
        
        # Blend with original image
        visualization = alpha * heatmap + (1 - alpha) * original_image
        
        return visualization.clamp(0, 1)


class AttentionPrior(nn.Module):
    """
    Improved attention prior that preserves spatial relationships
    Implements true spatial attention dynamics modeling
    """
    
    def __init__(
        self,
        attention_resolution: int = 21,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        motion_kernels: int = 8,
        use_separable_conv: bool = True
    ):
        super().__init__()
        self.attention_resolution = attention_resolution
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Spatial attention dynamics modeling
        if use_separable_conv:
            # More efficient separable convolutions
            self.spatial_dynamics = nn.Sequential(
                # Depthwise convolution (spatial processing)
                nn.Conv2d(1, 1, 5, padding=2, groups=1),
                nn.GroupNorm(1, 1),
                nn.SiLU(),
                # Pointwise convolution (channel mixing)
                nn.Conv2d(1, 16, 1),
                nn.GroupNorm(4, 16),
                nn.SiLU(),
                # Another depthwise for larger receptive field
                nn.Conv2d(16, 16, 5, padding=2, groups=16),
                nn.GroupNorm(4, 16),
                nn.SiLU(),
                nn.Conv2d(16, 32, 1),
                nn.GroupNorm(8, 32),
                nn.SiLU()
            )
        else:
            # Standard convolutions
            self.spatial_dynamics = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.GroupNorm(4, 16),
                nn.SiLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.SiLU()
            )
        
        # Motion prediction kernels (learned attention movement patterns)
        self.motion_kernels = nn.Parameter(
            torch.randn(motion_kernels, 1, 5, 5) * 0.01
        )
        
        # Context integration (h_t, z_t influence on spatial dynamics)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 32 + motion_kernels)  # For feature modulation
        )
        
        # Spatial-aware fusion
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(32 + motion_kernels, 16, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1)  # Output attention logits
        )
        
        # Optional: Attention movement predictor
        self.movement_predictor = nn.Conv2d(1, 2, 3, padding=1)  # Predicts dx, dy
    
    def compute_motion_features(
        self, 
        prev_attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract motion-relevant features from previous attention
        """
        batch_size = prev_attention.shape[0]
        prev_attention = prev_attention.unsqueeze(1)  # [B, 1, H, W]
        
        # Apply learned motion kernels
        motion_responses = []
        for kernel in self.motion_kernels:
            response = F.conv2d(
                prev_attention,
                kernel.unsqueeze(0),
                padding=2
            )
            motion_responses.append(response)
        
        motion_features = torch.cat(motion_responses, dim=1)  # [B, K, H, W]
        return motion_features
    
    def forward(
        self,
        prev_attention: torch.Tensor,  # [B, H, W]
        hidden_state: torch.Tensor,    # [B, hidden_dim]
        latent_state: torch.Tensor     # [B, latent_dim]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute spatially-aware attention prior
        """
        batch_size = prev_attention.shape[0]
        
        # 1. Extract spatial dynamics from previous attention
        spatial_features = self.spatial_dynamics(
            prev_attention.unsqueeze(1)
        )  # [B, 32, H, W]
        
        # 2. Compute motion features
        motion_features = self.compute_motion_features(prev_attention)
        
        # 3. Get context modulation
        context = torch.cat([hidden_state, latent_state], dim=-1)
        context_weights = self.context_projection(context)  # [B, 32 + K]
        
        # Split into spatial and motion weights
        spatial_weights, motion_weights = torch.split(
            context_weights, 
            [32, self.motion_kernels.shape[0]], 
            dim=-1
        )
        
        # 4. Modulate features with context
        # Reshape for spatial broadcasting
        spatial_weights = spatial_weights.view(batch_size, 32, 1, 1)
        motion_weights = motion_weights.view(batch_size, -1, 1, 1)
        
        modulated_spatial = spatial_features * torch.sigmoid(spatial_weights)
        modulated_motion = motion_features * torch.sigmoid(motion_weights)
        
        # 5. Combine and predict next attention
        combined_features = torch.cat([modulated_spatial, modulated_motion], dim=1)
        attention_logits = self.spatial_fusion(combined_features).squeeze(1)
        
        # 6. Apply softmax to get probabilities
        attention_probs = F.softmax(
            attention_logits.view(batch_size, -1), 
            dim=-1
        ).view(batch_size, self.attention_resolution, self.attention_resolution)
        
        # 7. Optional: Predict attention movement
        movement = self.movement_predictor(prev_attention.unsqueeze(1))
        dx, dy = movement[:, 0], movement[:, 1]
        
        return attention_probs, {
            'spatial_features': spatial_features,
            'motion_features': motion_features,
            'predicted_movement': (dx, dy),
            'attention_logits': attention_logits
        }

class LinearResidual(nn.Module):
    def __init__(
        self,
        input_feature: int,
        hidden_dim: int,
        *,
        nonlinearity=None,
        norm_type: str = "layer",   # 'layer' | 'batch' | None
        drop_p: float = 0.2         # dropout probability
    ):
        super().__init__()

        nl = nn.SiLU() if nonlinearity is None else nonlinearity
        self.residual_projection = (
            nn.Linear(input_feature, hidden_dim, bias=True)
            if input_feature != hidden_dim else nn.Identity()
        )

        layers = []
    
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(hidden_dim))

        layers.extend([nl, nn.Dropout(drop_p)])

        
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(hidden_dim))

        layers.extend([nl, nn.Dropout(drop_p)])

        self.fn = nn.Sequential(*layers)
        self.reset_parameters()     # custom init

    def reset_parameters(self):
        """
        Kaiming-uniform (fan_in) for all weight matrices (good for SiLU/ReLU).
        Last linear in the residual branch → weight = 0, bias = 0
          so the block begins as (roughly) the identity mapping.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last_linear = None
        for m in reversed(self.fn):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

        # If a projection exists, re-init it the same way
        if isinstance(self.residual_projection, nn.Linear):
            nn.init.kaiming_uniform_(
                self.residual_projection.weight, a=math.sqrt(5)
            )
            if self.residual_projection.bias is not None:
                nn.init.zeros_(self.residual_projection.bias)

    def forward(self, x):
        residual = self.residual_projection(x)
        return self.fn(residual) + residual


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
                    nn.SiLU(),
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
        self.layers = nn.ModuleList()
        
        # Initial layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True))
        )
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layer=nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            )
            if n==(n_layers-1):
               layer.append(attentionBlock(ndf* nf_mult))
            self.layers.append(layer)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            )
        )
        self.layers.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))

        # output 1 channel prediction map
        
        self.to(device)

    def get_features(self, x):
        """Extract features from intermediate layers"""
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, x, get_features=False):
        """Forward pass with option to return intermediate features"""
        if get_features:
            return self.get_features(x)
        
        # Regular forward pass
        for layer in self.layers:
            x = layer(x)
        return x


class LatentDiscriminator(nn.Module):
    # define the descriminator/critic
    def __init__(self, input_dims, num_layers=4, norm_type='layer', activation= nn.SiLU(), device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
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
                layers.append(LinearResidual(size // 2, size // 2, nonlinearity=self.activation, norm_type=self.norm_type))
            size = size // 2
        layers.append(nn.Linear(size, size * 2, bias=False))

        if self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(size * 2))
        elif self.norm_type == 'layer':
            layers.append(nn.LayerNorm(size * 2))

        # Activation Function
        layers.append(self.activation)
        # add anther residual block
        layers.append(LinearResidual(size * 2, size * 2, nonlinearity=self.activation, norm_type=self.norm_type))
        layers.append(nn.Linear(size * 2, 1))
        self.model = nn.Sequential(*layers)
        self.device = device
        self.to(device=self.device)

    def forward(self, x):
        return self.model(x)



class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with proper masking for temporal discrimination
    """
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1, 
                 sequence_length=128):
        super().__init__()
        assert n_embd % n_head == 0
        
        # Key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # Causal mask to ensure attention only attends to the left
        self.register_buffer("mask", 
            torch.tril(torch.ones(sequence_length, sequence_length))
            .view(1, 1, sequence_length, sequence_length))
        
        self.n_head = n_head
        self.n_embd = n_embd
        
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate query, key, values for all heads in batch
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        
        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        return y

class TransformerBlock(nn.Module):
    """Transformer block with causal attention"""
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1,
                 sequence_length=128):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop,
                                        sequence_length)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        m = self.mlp
        x = x + m.dropout(m.c_proj(m.act(m.c_fc(self.ln_2(x)))))
        return x

class TemporalDiscriminator(nn.Module):
    """
    Temporal Discriminator using Causal Transformer
    
    Evaluates sequence quality by processing temporal relationships
    """
    def __init__(
        self,
        input_channels: int = 3,
        image_size: int = 64,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        sequence_length: int = 16,
        feature_extractor_channels: List[int] = [64, 128, 256, 512],
        use_spectral_norm: bool = True,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Frame-level feature extractor (CNN)
        self.frame_encoder = self._build_frame_encoder(
            input_channels, feature_extractor_channels, use_spectral_norm
        )
        
        # Calculate feature dimension after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, image_size, image_size)
            dummy_features = self.frame_encoder(dummy_input)
            feature_dim = dummy_features.shape[1]
        
        # Project CNN features to transformer dimension
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, sequence_length, hidden_dim)
        )
        self.drop = nn.Dropout(0.1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim, n_heads, 
                sequence_length=sequence_length
            ) for _ in range(n_layers)
        ])
        
        # Final normalization
        self.ln_f = nn.LayerNorm(hidden_dim)
        
        # Discrimination heads
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.per_frame_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        self.to(device)
        
    def _build_frame_encoder(self, input_channels, channels, use_spectral_norm):
        """Build CNN for frame-level feature extraction"""
        layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.extend([
                conv,
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def extract_features(self, x):
        """
        Extract features from sequence of frames
        
        Args:
            x: [batch_size, seq_len, channels, height, width]
        
        Returns:
            features: [batch_size, seq_len, hidden_dim]
        """
        B, T, C, H, W = x.shape
        
        # Process all frames through CNN
        x_flat = x.view(B * T, C, H, W)
        features_flat = self.frame_encoder(x_flat)
        features = features_flat.view(B, T, -1)
        
        # Project to transformer dimension
        features = self.feature_projection(features)
        
        return features
    
    def forward(self, x, return_features=False):
        """
        Forward pass through temporal discriminator
        
        Args:
            x: [batch_size, seq_len, channels, height, width]
            return_features: If True, return intermediate features
        
        Returns:
            dict with discrimination scores and optionally features
        """
        B, T, C, H, W = x.shape
        assert T <= self.sequence_length, f"Sequence length {T} exceeds maximum {self.sequence_length}"
        
        # Extract frame features
        features = self.extract_features(x)
        
        # Add positional embeddings
        features = features + self.pos_embed[:, :T, :]
        features = self.drop(features)
        
        # Process through transformer
        hidden = features
        all_hidden_states = []
        
        for block in self.blocks:
            hidden = block(hidden)
            if return_features:
                all_hidden_states.append(hidden)
        
        hidden = self.ln_f(hidden)
        
        # Get discrimination scores
        # 1. Temporal coherence score (from last timestep)
        temporal_score = self.temporal_head(hidden[:, -1, :])
        
        # 2. Per-frame quality scores
        per_frame_scores = self.per_frame_head(hidden)
        
        outputs = {
            'temporal_score': temporal_score,  # [B, 1]
            'per_frame_scores': per_frame_scores,  # [B, T, 1]
            'final_score': temporal_score + per_frame_scores.mean(dim=1)  # Combined
        }
        
        if return_features:
            outputs['features'] = all_hidden_states
            outputs['frame_features'] = features
        
        return outputs

class TemporalLatentDiscriminator(nn.Module):
    """
    Temporal Discriminator for Latent Sequences using Self-Attention
    
    Processes latent sequences to evaluate temporal coherence and quality
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        sequence_length: int = 16,
        use_spectral_norm: bool = True,
        dropout: float = 0.1,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.device = device
        
        # Project latent to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        if use_spectral_norm:
            self.input_projection[0] = nn.utils.spectral_norm(self.input_projection[0])
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, sequence_length, hidden_dim)
        )
        
        # Temporal processing blocks
        self.temporal_blocks = nn.ModuleList()
        for _ in range(n_layers):
            # Self-attention block
            self.temporal_blocks.append(
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(
                        hidden_dim, n_heads, 
                        dropout=dropout, 
                        batch_first=True
                    ),
                    'norm1': nn.LayerNorm(hidden_dim),
                    'ffn': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 4, hidden_dim),
                        nn.Dropout(dropout)
                    ),
                    'norm2': nn.LayerNorm(hidden_dim)
                })
            )
        
        # Output heads
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sequence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Contrastive head for temporal consistency
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(device)
    
    def compute_temporal_features(self, x):
        """
        Extract temporal features using self-attention
        
        Args:
            x: [batch_size, seq_len, latent_dim]
        
        Returns:
            features: [batch_size, seq_len, hidden_dim]
        """
        B, T, D = x.shape
        
        # Project to hidden dimension
        h = self.input_projection(x)
        
        # Add positional encoding
        h = h + self.pos_encoding[:, :T, :]
        
        # Process through temporal blocks
        for block in self.temporal_blocks:
            # Self-attention
            h_norm = block['norm1'](h)
            attn_out, _ = block['attention'](h_norm, h_norm, h_norm)
            h = h + attn_out
            
            # FFN
            h_norm = block['norm2'](h)
            h = h + block['ffn'](h_norm)
        
        return h
    
    def forward(self, x, return_features=False):
        """
        Forward pass through temporal latent discriminator
        
        Args:
            x: [batch_size, seq_len, latent_dim]
            return_features: If True, return intermediate features
        
        Returns:
            Dictionary with discrimination scores
        """
        B, T, D = x.shape
        assert T <= self.sequence_length, f"Sequence length {T} exceeds maximum {self.sequence_length}"
        
        # Extract temporal features
        features = self.compute_temporal_features(x)
        
        # Temporal coherence score (from last timestep)
        temporal_score = self.temporal_head(features[:, -1, :])
        
        # Sequence quality score (average pooled)
        sequence_score = self.sequence_head(features.mean(dim=1))
        
        # Temporal consistency score
        # Compare adjacent timesteps
        if T > 1:
            # Compute differences between adjacent timesteps
            diffs = features[:, 1:, :] - features[:, :-1, :]
            diff_features = torch.cat([
                diffs.mean(dim=1),  # Average temporal difference
                diffs.std(dim=1)    # Temporal variance
            ], dim=-1)
            consistency_score = self.consistency_head(diff_features)
        else:
            consistency_score = torch.zeros(B, 1).to(x.device)
        
        outputs = {
            'temporal_score': temporal_score,
            'sequence_score': sequence_score,
            'consistency_score': consistency_score,
            'final_score': temporal_score + sequence_score + 0.5 * consistency_score
        }
        
        if return_features:
            outputs['features'] = features

        return outputs

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
                
    @torch.compile  # Add torch.compile for faster inference
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def compute_feature_matching_loss( real_features, fake_features, lambda_feat=1.0):
    """Compute feature matching loss between real and fake image features"""
    feat_match_loss = 0.0
    num_layers = len(real_features)
    
    # Match mean activations for each layer
    for i in range(num_layers):
        real_mean = real_features[i].mean(dim=[0,2,3]) # Average across batch and spatial dims
        fake_mean = fake_features[i].mean(dim=[0,2,3])
        feat_match_loss += torch.nn.functional.l1_loss(fake_mean, real_mean.detach())
        
    return lambda_feat * feat_match_loss / num_layers
