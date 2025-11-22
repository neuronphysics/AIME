import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Optional, Tuple
from contextlib import contextmanager
import abc
import math
from collections import OrderedDict

def swish(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation layer from NVAE"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return x * self.se(x)



def spectral_norm_conv(conv_layer):
    """Apply spectral normalization to conv layers"""
    return nn.utils.spectral_norm(conv_layer)


class NVAEResidualBlock(nn.Module):
    """NVAE-style residual block with SE and BN"""
    def __init__(
        self,
        in_channels,
        out_channels=None,
        stride=1,
        use_se=True,
        norm_type='group',  # 'batch' or 'group'
        dropout=0.0
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        # Choose normalization
        if norm_type == 'batch':
            Norm = nn.BatchNorm2d
        else:
            def Norm(channels):
                return nn.GroupNorm(min(32, channels), channels)
        
        # Main path
        self.conv1 = spectral_norm_conv(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        )
        self.norm1 = Norm(out_channels)
        
        self.conv2 = spectral_norm_conv(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        )
        self.norm2 = Norm(out_channels)
        
        # Squeeze-Excitation
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        
        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = spectral_norm_conv(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.gate = nn.Parameter(torch.zeros(1))  # Learnable gate for residual
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = swish(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = swish(out)
        
        # Gated residual connection
        out = identity + self.gate * out
        return out


class EncoderResidualBlock(nn.Module):
    """Encoder block with optional downsampling"""
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        num_blocks=2,
        use_se=True,
        dropout=0.0
    ):
        super().__init__()
        blocks = []
        
        # First block handles stride
        blocks.append(
            NVAEResidualBlock(in_channels, out_channels, stride, use_se, dropout=dropout)
        )
        
        # Additional blocks
        for _ in range(num_blocks - 1):
            blocks.append(
                NVAEResidualBlock(out_channels, out_channels, 1, use_se, dropout=dropout)
            )
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)


class DecoderResidualBlock(nn.Module):
    """Decoder block with optional upsampling using nearest neighbor + conv"""
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample=False,
        num_blocks=2,
        use_se=True,
        dropout=0.0
    ):
        super().__init__()
        blocks = []
        
        # Upsampling if needed (nearest neighbor + conv to avoid checkerboard)
        if upsample:
            blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
            blocks.append(
                spectral_norm_conv(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
                )
            )
            #blocks.append(nn.BatchNorm2d(out_channels))
            num_groups = min(32, out_channels) if out_channels % 8 == 0 else out_channels // 4 if out_channels > 4 else 1
            blocks.append(nn.GroupNorm(num_groups, out_channels))
 
            blocks.append(Swish())
            current_channels = out_channels
        else:
            current_channels = in_channels
        
        # Residual blocks
        for i in range(num_blocks):
            in_ch = current_channels if i == 0 else out_channels
            blocks.append(
                NVAEResidualBlock(in_ch, out_channels, 1, use_se, dropout=dropout)
            )
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)



class VAEEncoder(nn.Module):
    """NVAE-style encoder with progressive downsampling"""
    def __init__(
        self,
        channel_size_per_layer: List[int],
        layers_per_block_per_layer: List[int],
        latent_size: int,
        width: int,
        height: int,
        num_layers_per_resolution: List[int],
        mlp_hidden_size: int = 512,
        channel_size: int = 64,
        input_channels: int = 3,
        downsample: int = 4,
        use_se: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.latent_size = latent_size
        self.downsample = downsample
        self._cached_levels = None   # dict like {"C2": Tensor, ...}
        self._cached_hw = None       # dict like {"H×W": Tensor, ...}
        self._forward_count = 0      # 

        if channel_size % 32 == 0:
            num_groups = 32
        elif channel_size % 16 == 0:
            num_groups = 16
        elif channel_size % 8 == 0:
            num_groups = 8
        elif channel_size % 4 == 0:
            num_groups = 4
        else:
            num_groups = 1  # Fall back to LayerNorm equivalent

        # Initial convolution
        self.stem = nn.Sequential(
            spectral_norm_conv(
                nn.Conv2d(input_channels, channel_size, 7, padding=3, stride=1, bias=False)
            ),
            nn.GroupNorm(num_groups, channel_size),
            Swish()
        )
        
        # Build encoder blocks with progressive downsampling
        encoder_blocks = []
        current_channels = channel_size
        
        layer_idx = 0
        for resolution_idx, num_layers in enumerate(num_layers_per_resolution):
            # Determine if we should downsample at this resolution
            should_downsample = resolution_idx < downsample
            
            for block_idx in range(num_layers):
                out_channels = channel_size_per_layer[layer_idx]
                num_residual_blocks = layers_per_block_per_layer[layer_idx]
                
                # Only downsample on first block of each resolution level
                stride = 2 if (should_downsample and block_idx == 0) else 1
                
                encoder_blocks.append(
                    EncoderResidualBlock(
                        current_channels,
                        out_channels,
                        stride=stride,
                        num_blocks=num_residual_blocks,
                        use_se=use_se,
                        dropout=dropout
                    )
                )
                current_channels = out_channels
                layer_idx = layer_idx + 1

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        
        # Final processing
        self.final_norm = nn.BatchNorm2d(current_channels)
        self.final_act = Swish()
        
        # Compute final spatial dimensions
        final_h = height // (2 ** downsample)
        final_w = width // (2 ** downsample)
        
        # MLP for latent projection
        mlp_input_size = current_channels * final_h * final_w
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.LayerNorm(mlp_hidden_size),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.LayerNorm(mlp_hidden_size),
            Swish(),
        )
        
        # Separate projections for mean and log variance
        self.proj = nn.Linear(mlp_hidden_size, 2*latent_size)
        self.use_checkpoint = False
        
    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing.
        """
        self.use_checkpoint = False

    def gradient_checkpointing_enable(self):

        self.use_checkpoint = True

    @contextmanager
    def enable_caching(self):
        """
        Context manager that enables one-pass feature caching for decoder/aux losses.
        """
        try:
            self._cached_levels = {}
            self._cached_hw = {}
            yield
        finally:
            self._cached_levels = None
            self._cached_hw = None

    
    def forward(self, x):
        # Stem
        h = self.stem(x)
        stem_out = h  # for potential skip connections
        
        # Progressive encoding
        skip_connections = []
        for block in self.encoder_blocks:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False, preserve_rng_state=True)
            else:
                h = block(h)
            skip_connections.append(h)

        if self._cached_levels is not None and self._cached_hw is not None:
            # Mirror extract_pyramid's level semantics
            prev_hw = stem_out.shape[-2:]  # resolution right after stem
            level = 1
            self._cached_levels["C1"] = stem_out
            self._cached_hw[f"{prev_hw[0]}x{prev_hw[1]}"] = stem_out

            for feat in skip_connections:
                hw = feat.shape[-2:]
                if hw != prev_hw:
                    level += 1
                    prev_hw = hw
                # last block at a resolution wins (same as extract_pyramid)
                self._cached_levels[f"C{level}"] = feat
                self._cached_hw[f"{hw[0]}x{hw[1]}"] = feat
        
        # Final processing
        h = self.final_norm(h)
        h = self.final_act(h)
        
        # MLP
        if self.use_checkpoint and self.training:
            # Checkpoint the MLP computation
            hlayer = torch.utils.checkpoint.checkpoint(
                self.mlp, h, use_reentrant=False, preserve_rng_state=True
            )
            # Checkpoint the projection to mean/logvar
            mean_logvar = torch.utils.checkpoint.checkpoint(
                self.proj, hlayer, use_reentrant=False, preserve_rng_state=True
            )
        else:
            hlayer = self.mlp(h)
            mean_logvar = self.proj(hlayer)

        mean, logvar = mean_logvar.chunk(2, dim=-1)
        # Project to latent space
        logvar = torch.clamp(logvar, -10.0, 2.0)  # numerical stability
        sigma = (logvar * 0.5).exp()
        eps= torch.randn_like(sigma)
        z = mean + eps * sigma
        return z, mean, logvar, hlayer
    
    def extract_pyramid(self, x: torch.Tensor, levels=None):
        """Return dict of multi-scale features tapped from the encoder backbone.
        Levels can be an int (keep last N scales) or an iterable of names (e.g., ("C2","C3","C4")).
        This method *does not* change the encoder's latent path.
        """
        feats = {}
        cur = x
        level = 0
        prev_hw = cur.shape[-2:]
        if hasattr(self, 'stem') and callable(getattr(self, 'stem')):
            cur = self.stem(x)
            prev_hw = cur.shape[-2:]
            feats[f"C{level+1}"] = cur
        for block in self.encoder_blocks:
            cur = block(cur)
            hw = cur.shape[-2:]
            if hw != prev_hw:
                level = level + 1
                prev_hw = hw
            feats[f"C{level+1}"] = cur
        if levels is None:
            keys = sorted(feats.keys(), key=lambda k: int(k[1:]))
            sel = keys[-3:]
        elif isinstance(levels, int):
            keys = sorted(feats.keys(), key=lambda k: int(k[1:]))
            sel = keys[-levels:]
        else:
            sel = [k for k in levels if k in feats]
        return {k: feats[k] for k in sel}

    def get_unet_skips(self, x, levels=None, detach: bool = False):
        if x is None and (self._cached_levels is not None) and (self._cached_hw is not None):
            # Decide which levels to serve (mirror extract_pyramid)
            if levels is None:
                keys = sorted(self._cached_levels.keys(), key=lambda k: int(k[1:]))[-3:]
            elif isinstance(levels, int):
                keys = sorted(self._cached_levels.keys(), key=lambda k: int(k[1:]))[-levels:]
            else:
                keys = [k for k in levels if k in self._cached_levels]

            out = {}
            for k in keys:
                t = self._cached_levels[k]
                t = t.detach() if detach else t
                H, W = t.shape[-2:]
                out[f"{H}x{W}"] = t
            return out

        feats = self.extract_pyramid(x, levels=levels)
        out = {}
        for k, v in feats.items():
            t = v.detach() if detach else v
            H, W = t.shape[-2:]
            out[f"{H}x{W}"] = t
        return out

    def build_attention_fuser(self, 
                              sample_input: torch.Tensor,
                              out_hw=(21, 21),
                              source=None,
                              d: int = 64):
        """
        Prebuild the 1x1 projection used by fused_attention_features(..., fuse='concat+1x1').
        """
       
        with torch.no_grad():
            feats = self.extract_pyramid(sample_input, levels=source)  
            ups = [F.interpolate(f, size=out_hw, mode='bilinear', align_corners=False) for f in feats.values()]
            fused = torch.cat(ups, dim=1)                               # C_in = sum(C_level)
            in_ch = fused.shape[1]

        # Create or replace 1x1 projection
        self.attn_fuse_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=True).to(sample_input.device)


    def fused_attention_features(self, 
                                 x: torch.Tensor, 
                                 out_hw=(21, 21),
                                 source=None, 
                                 fuse='concat', 
                                 d=None,
                                 detach=False,
                                 ):
        """Produce a single fused feature map for the AttentionPosterior.
        See README for details."""
        feats = self.extract_pyramid(x, levels=source)
        ups = []
        for k in feats:
            t = feats[k].detach() if detach else feats[k]
            t = F.interpolate(t, size=out_hw, mode='bilinear', align_corners=False)
            ups.append(t)
        fused = torch.cat(ups, dim=1)
        if fuse == 'concat':
            return fused
        elif fuse == 'concat+1x1':
            
            return self.attn_fuse_proj(fused)
        else:
            raise ValueError(f"Unknown fuse mode: {fuse}")
            
# Mixture of Discretized Logistics 
class MDLHead(nn.Module):
    """RGB mixture of discretized logistics (PixelCNN++ style)."""
    def __init__(self, in_ch, out_ch, n_mix=10):
        super().__init__()
        self.n_mix = n_mix
        self.out_ch = out_ch
        self.use_checkpoint = False
        if out_ch != 3:
            self.params_pre_mix =1+2*out_ch
        else:
            self.params_pre_mix = 10 # 1 logit + 3 means + 3 log_scales + 3 coeffs
        # logits + means(3) + log_scales(3) + coeffs(3) = 10 per mixture
        self.proj = nn.Conv2d(in_ch, n_mix * self.params_pre_mix, 1, bias=True)

    def gradient_checkpointing_enable(self): 

        self.use_checkpoint = True
    def gradient_checkpointing_disable(self): 

        self.use_checkpoint = False

    def _split(self, y):
        B, _, H, W = y.shape
        if self.out_ch == 3:
            y = y.view(B, self.n_mix, 10, H, W)
            logit_probs = y[:, :, 0]                 # [B,K,H,W]
            means       = y[:, :, 1:4]               # [B,K,3,H,W]
            log_scales  = y[:, :, 4:7].clamp(min=-7) # [B,K,3,H,W]
            coeffs      = torch.tanh(y[:, :, 7:10])  # [B,K,3,H,W]
            return logit_probs, means, log_scales, coeffs
        else:
            y = y.view(B, self.n_mix, self.params_pre_mix, H, W)
            logit_probs = y[:, :, 0]                 # [B,K,H,W]
            means       = y[:, :, 1:1+self.out_ch]               # [B,K,C,H,W]
            log_scales  = y[:, :, 1+self.out_ch:1+2*self.out_ch].clamp(min=-7) # [B,K,C,H,W]
            return logit_probs, means, log_scales, None
    
    def forward(self, h):
        return self._split(self.proj(h))
    
    @torch.no_grad()
    def sample(self, logit_probs, means, log_scales, coeffs,
            scale_temp: float = 1.0, mode: str = "sample"):

        B, K, H, W = logit_probs.shape

        # one helper for both branches
        def samp(mu, log_s):
            u = torch.rand_like(mu).clamp(1e-5, 1 - 1e-5)
            return mu + torch.exp(log_s) * (torch.log(u) - torch.log1p(-u))

        # pick mixture component per pixel
        mix = torch.distributions.Categorical(logits=logit_probs.permute(0, 2, 3, 1)).sample()
        sel = mix.unsqueeze(1).unsqueeze(1)  # shape align for gather

        if self.out_ch == 3:
            # gather selected component params for RGB
            m  = means.gather(1, sel.expand(-1, 1, 3, -1, -1)).squeeze(1)       # [B,3,H,W]
            ls = log_scales.gather(1, sel.expand(-1, 1, 3, -1, -1)).squeeze(1)   # [B,3,H,W]
            cs = coeffs.gather(1, sel.expand(-1, 1, 3, -1, -1)).squeeze(1)       # [B,3,H,W]

            if mode == "mean":
                # deterministic RGB with PixelCNN++-style linear couplings
                x_r = m[:, 0]
                x_g = m[:, 1] + cs[:, 0] * x_r
                x_b = m[:, 2] + cs[:, 1] * x_r + cs[:, 2] * x_g
                x = torch.stack([x_r, x_g, x_b], dim=1)
                return x.clamp(-1, 1)

            # shrink scales for sampling
            ls = ls + math.log(max(scale_temp, 1e-6))

            # sample with couplings
            x_r = samp(m[:, 0], ls[:, 0])
            x_g = samp(m[:, 1] + cs[:, 0] * x_r, ls[:, 1])
            x_b = samp(m[:, 2] + cs[:, 1] * x_r + cs[:, 2] * x_g, ls[:, 2])
            x = torch.stack([x_r, x_g, x_b], dim=1)

        else:
            # generic C-channel case (no cross-channel coupling)
            C = self.out_ch
            m  = means.gather(1, sel.expand(-1, 1, C, -1, -1)).squeeze(1)        # [B,C,H,W]
            ls = log_scales.gather(1, sel.expand(-1, 1, C, -1, -1)).squeeze(1)   # [B,C,H,W]

            if mode == "mean":
                return m.clamp(-1, 1)

            ls = ls + math.log(max(scale_temp, 1e-6))
            x  = samp(m, ls)  # elementwise

        return x.clamp(-1, 1)
    
    def nll(self, x, logit_probs, means, log_scales, coeffs):
        """
        x: [B,3,H,W] in [-1,1]
        returns per-image NLL (mean over pixels & channels).
        """
        def _nll_core(x, logit_probs, means, log_scales, coeffs):
            B, _, H, W = x.shape
            K = self.n_mix
            n_bits=8
            bin_size = 1/(2**n_bits-1)
            
            # Expand x to [B,K,3,H,W]
            xk = x.unsqueeze(1).expand(-1, K, -1, -1, -1)
            
            if self.out_ch == 3:
                # Coupled means per channel
                mu_r = means[:, :, 0]
                mu_g = means[:, :, 1] + coeffs[:, :, 0] * xk[:, :, 0]
                mu_b = means[:, :, 2] + coeffs[:, :, 1] * xk[:, :, 0] + coeffs[:, :, 2] * xk[:, :, 1]
                mu = torch.stack([mu_r, mu_g, mu_b], dim=2)          # [B,K,3,H,W]
            else:
                mu = means                                            # [B,K,C,H,W]

            inv_s = torch.exp(-log_scales)                            # [B,K,C,H,W]

            # CDF at bin edges
            x_lo = (xk - bin_size/2 - mu) * inv_s
            x_hi = (xk + bin_size/2 - mu) * inv_s
            cdf_lo = torch.sigmoid(x_lo)
            cdf_hi = torch.sigmoid(x_hi)

            # Edge bins
            log_cdf_hi = (cdf_hi.clamp_min(1e-12)).log()
            log_one_minus_cdf_lo = (1 - cdf_lo).clamp_min(1e-12).log()

            # Discretized logistic mass
            p = torch.where(
                xk < -0.999,                       # near -1
                torch.exp(log_cdf_hi),
                torch.where(
                    xk > 0.999,                    # near +1
                    torch.exp(log_one_minus_cdf_lo),
                    (cdf_hi - cdf_lo).clamp_min(1e-12)
                )
            )

            # Mix across channels, then across components
            log_p   = p.clamp_min(1e-12).log().sum(dim=2)            # [B,K,H,W]
            log_mix = log_p + F.log_softmax(logit_probs, dim=1)      # [B,K,H,W]
            log_px  = torch.logsumexp(log_mix, dim=1)                # [B,H,W]

            # Mean NLL per image
            return -log_px.mean(dim=(1,2))                           # [B]

        if self.use_checkpoint and self.training:
            # checkpoint recomputes _nll_core in backward; at least one arg must require grad
            return torch.utils.checkpoint.checkpoint(
                _nll_core, x, logit_probs, means, log_scales, 
                torch.zeros(1, device=x.device) if coeffs is None else coeffs,
                use_reentrant=False, preserve_rng_state=True
            )
        else:
            return _nll_core(x, logit_probs, means, log_scales, coeffs)

class VAEDecoder(nn.Module):
    """NVAE-style decoder with nearest neighbor upsampling"""
    def __init__(
        self,
        latent_size: int,
        width: int,
        height: int,
        channel_size_per_layer: List[int] = (256, 256, 256, 256, 128, 128, 64, 64),
        layers_per_block_per_layer: List[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        num_layers_per_resolution: List[int] = (2, 2, 2, 2),
        reconstruction_channels: int = 3,
        downsample: Optional[int] = 4,
        mlp_hidden_size: Optional[int] = 512,
        use_se: bool = True,
        dropout: float = 0.0,
        unet_use_checkpoint: bool = False,
        encoder_channel_size_per_layer: Optional[List[int]] = None,
        skip_levels_last_n: int = 3,
    ):
        super().__init__()
        self.downsample = downsample
        self.use_checkpoint : bool = unet_use_checkpoint
        # Compute initial spatial dimensions
        initial_h = height // (2 ** downsample)
        initial_w = width // (2 ** downsample)
        initial_channels = channel_size_per_layer[0]
        # Store config for UNet adapter construction
        self._unet_height = height
        self._unet_width = width
        self._unet_channel_size_per_layer = list(channel_size_per_layer)
        self._unet_num_layers_per_resolution = list(num_layers_per_resolution)
        self._unet_downsample = downsample if downsample is not None else 0

        # MLP to project from latent to spatial
        mlp_output_size = initial_channels * initial_h * initial_w
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, mlp_hidden_size),
            nn.LayerNorm(mlp_hidden_size),  
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.LayerNorm(mlp_hidden_size),  
            Swish(),
            nn.Linear(mlp_hidden_size, mlp_output_size),
            nn.LayerNorm(mlp_output_size),  
            Swish(),
        )
        
        self.unflatten = nn.Unflatten(1, (initial_channels, initial_h, initial_w))
        
        # Build decoder blocks with progressive upsampling
        decoder_blocks = []
        current_channels = initial_channels
        
        layer_idx = 0
        upsample_count = 0
        
        for resolution_idx, num_layers in enumerate(num_layers_per_resolution):
            # Determine if we should upsample at this resolution
            should_upsample = (resolution_idx >= len(num_layers_per_resolution) - downsample 
                             and upsample_count < downsample)
            
            for block_idx in range(num_layers):
                out_channels = channel_size_per_layer[layer_idx]
                num_residual_blocks = layers_per_block_per_layer[layer_idx]
                
                # Only upsample on first block of each resolution level
                upsample = should_upsample and block_idx == 0
                if upsample:
                    upsample_count = upsample_count + 1

                decoder_blocks.append(
                    DecoderResidualBlock(
                        current_channels,
                        out_channels,
                        upsample=upsample,
                        num_blocks=num_residual_blocks,
                        use_se=use_se,
                        dropout=dropout
                    )
                )
                current_channels = out_channels
                layer_idx = layer_idx + 1
        self._unet_encoder_channel_size_per_layer = (
            encoder_channel_size_per_layer
            if encoder_channel_size_per_layer is not None
            else channel_size_per_layer   # fallback: symmetric encoder/decoder
        )
        self._unet_skip_levels_last_n = skip_levels_last_n

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        # Final layers
        #self.final_norm = nn.BatchNorm2d(current_channels)
        num_groups = min(32, current_channels) if current_channels % 8 == 0 else current_channels // 4 if current_channels > 4 else 1
        self.final_norm = nn.GroupNorm(num_groups, current_channels)

        self.final_act = Swish()
        
        self.pre_output = nn.Sequential(
                        nn.Conv2d(current_channels, current_channels // 2, 1, bias=True),
                        nn.SiLU(),
                    )
                    # MDL head
        self.mdl_head = MDLHead(current_channels // 2, out_ch=reconstruction_channels, n_mix=10)
        if hasattr(self.mdl_head, 'gradient_checkpointing_enable'):
           self.mdl_head.gradient_checkpointing_enable()
        
        
        self._pending_skips: Optional[dict]= None               # dict[str, Tensor], set via set_unet_skips(...)
        self._unet_mode: str = "concat"          
        self.post_concat_adapters = nn.ModuleDict()  # 1x1 convs to map cat(C_in+skip) -> C_in
        if self._unet_downsample is not None and self._unet_downsample > 0:
            self._init_unet_post_concat_adapters(
                height=height,
                width=width,
                encoder_channel_size_per_layer=self._unet_encoder_channel_size_per_layer,
                decoder_channel_size_per_layer=self._unet_channel_size_per_layer,
                num_layers_per_resolution=self._unet_num_layers_per_resolution,
                downsample=self._unet_downsample,
                skip_levels_last_n=self._unet_skip_levels_last_n,
            )

    def gradient_checkpointing_disable(self):

        self.use_checkpoint = False

    def gradient_checkpointing_enable(self):

        self.use_checkpoint = True

    def _init_unet_post_concat_adapters(
        self,
        height: int,
        width: int,
        encoder_channel_size_per_layer: List[int],
        decoder_channel_size_per_layer: List[int],
        num_layers_per_resolution: List[int],
        downsample: int,
        skip_levels_last_n: int = 3,
    ) -> None:
        """
        Analytically mirror the VAEEncoder / VAEDecoder schedule to figure out
        which (in_ch, out_ch, H, W) combos will be used for UNet skip fusion,
        and create a Conv2d for each of them in self.post_concat_adapters.

        Encoder and decoder can have different channel schedules:
          * encoder_channel_size_per_layer: used to infer skip channels per level
          * decoder_channel_size_per_layer: used to follow the actual decoder blocks
        """

        if downsample is None or downsample <= 0:
            # No spatial down/up-sampling => no UNet-style fusion
            return

        # ---- 1) Simulate encoder pyramid levels (C1..C_{downsample+1}) ----
        # enc_levels[level_idx] = (H, W, C)
        enc_levels = {}
        cur_h, cur_w = height, width
        level = 0
        enc_levels[1] = (cur_h, cur_w, None)  # C1; channels don't matter here
        layer_idx = 0

        for resolution_idx, n_layers in enumerate(num_layers_per_resolution):
            should_downsample = resolution_idx < downsample
            for block_idx in range(n_layers):
                stride = 2 if (should_downsample and block_idx == 0) else 1
                if stride == 2:
                    # spatial downsample in encoder
                    cur_h //= 2
                    cur_w //= 2
                    level += 1
                out_ch = encoder_channel_size_per_layer[layer_idx]
                layer_idx += 1
                enc_levels[level + 1] = (cur_h, cur_w, out_ch)

        max_level = max(enc_levels.keys())
        skip_levels_last_n = min(skip_levels_last_n, max_level)

        # Mimic get_unet_skips(levels=None): take last N pyramid levels
        selected_levels = list(
            range(max_level - skip_levels_last_n + 1, max_level + 1)
        )
        from collections import OrderedDict
        skip_pool_template = OrderedDict()
        for L in selected_levels:
            H, W, C = enc_levels[L]
            skip_pool_template[f"{H}x{W}"] = C  # same keys as get_unet_skips

        # ---- 2) Simulate decoder blocks & where fusion actually happens ----
        init_h = height // (2 ** downsample)
        init_w = width // (2 ** downsample)
        cur_h, cur_w = init_h, init_w
        cur_ch = decoder_channel_size_per_layer[0]  # initial decoder channels

        layer_idx = 0
        upsample_count = 0
        adapter_shapes = []

        # Copy so we don't mutate the template
        pool = OrderedDict(skip_pool_template)

        for resolution_idx, n_layers in enumerate(num_layers_per_resolution):
            # Same condition as in __init__
            should_upsample = (
                resolution_idx >= len(num_layers_per_resolution) - downsample
                and upsample_count < downsample
            )
            for block_idx in range(n_layers):
                prev_hw = (cur_h, cur_w)
                upsample = bool(should_upsample and block_idx == 0)
                if upsample:
                    upsample_count += 1
                    cur_h *= 2
                    cur_w *= 2

                out_ch = decoder_channel_size_per_layer[layer_idx]
                layer_idx += 1
                cur_ch = out_ch
                new_hw = (cur_h, cur_w)

                # Fusion triggers only when spatial size changed and we still have skips.
                if pool and new_hw != prev_hw:
                    H, W = new_hw
                    hw_key = f"{H}x{W}"

                    if hw_key in pool:
                        # exact spatial match
                        skip_ch = pool.pop(hw_key)
                    else:
                        # Fallback: same as _pick_and_prep_skip_from -> popitem() (LIFO)
                        _, skip_ch = pool.popitem()

                    in_ch = cur_ch + skip_ch
                    adapter_shapes.append((in_ch, cur_ch, H, W))

        # ---- 3) Instantiate one Conv2d per unique (in_ch, out_ch, H, W) ----
        seen = set()
        for in_ch, out_ch, H, W in adapter_shapes:
            key = f"cat_{in_ch}->{out_ch}@{H}x{W}"
            if key in seen:
                continue
            seen.add(key)
            self.post_concat_adapters[key] = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, bias=True
            )
            print(f"Created UNet adapter: {key}")

    def build_unet_adapters_from_skips(self, skips: dict) -> None:
        """
        Prebuild self.post_concat_adapters using actual encoder skip tensors.

        Call this once during model initialization (e.g. from
        DPGMMVariationalRecurrentAutoencoder._warm_build_unet_adapters),
        before creating the optimizer. After this, _fuse_unet_skip_if_needed
        will never instantiate new layers.
        """

        if not skips:
            return

        downsample = int(self._unet_downsample)
        if downsample <= 0:
            return

        # 1) Get skip channels per spatial resolution, preserving encoder order
        pool = OrderedDict()
        for k, v in skips.items():        # k like "32x32"
            H, W = map(int, k.split("x"))
            pool[(H, W)] = v.shape[1]     # channels

        # 2) Mirror the decoder schedule (same logic as in __init__ / forward)
        cur_h = self._unet_height // (2 ** downsample)
        cur_w = self._unet_width  // (2 ** downsample)
        cur_ch = self._unet_channel_size_per_layer[0]
        layer_idx = 0
        upsample_count = 0

        # Use a reference param to put new convs on the right device / dtype
        ref_param = next(self.parameters())
        device = ref_param.device
        dtype  = ref_param.dtype

        for resolution_idx, n_layers in enumerate(self._unet_num_layers_per_resolution):
            should_upsample = (
                resolution_idx >= len(self._unet_num_layers_per_resolution) - downsample
                and upsample_count < downsample
            )
            for block_idx in range(n_layers):
                prev_hw = (cur_h, cur_w)
                upsample = bool(should_upsample and block_idx == 0)
                if upsample:
                    upsample_count += 1
                    cur_h *= 2
                    cur_w *= 2

                out_ch = self._unet_channel_size_per_layer[layer_idx]
                layer_idx += 1
                cur_ch = out_ch
                new_hw = (cur_h, cur_w)

                # We only fuse when the spatial size changed and there are skips left
                if pool and new_hw != prev_hw:
                    H, W = new_hw
                    key_exact = (H, W)

                    if key_exact in pool:
                        skip_ch = pool.pop(key_exact)
                    else:
                        match_key = None
                        for hw in pool.keys():
                            if hw == (H, W):
                                match_key = hw
                                break
                        if match_key is not None:
                            skip_ch = pool.pop(match_key)
                        else:
                            # Same fallback as _pick_and_prep_skip_from: last inserted
                            _, skip_ch = pool.popitem()

                    in_ch = cur_ch + skip_ch
                    adapter_key = f"cat_{in_ch}->{cur_ch}@{H}x{W}"
                    if adapter_key not in self.post_concat_adapters:
                        self.post_concat_adapters[adapter_key] = nn.Conv2d(
                            in_ch, cur_ch, kernel_size=1, bias=True
                        ).to(device=device, dtype=dtype)

    def set_unet_skips(self, skips: Optional[dict], mode: str = "concat"):
        """
        Provide encoder feature maps for UNet-style fusion. Call right before forward(z).

        """
        
        self._unet_mode = mode
        # Store as a plain dict; do NOT consume/mutate this inside forward (checkpoint-safety).
        self._pending_skips = dict(skips) if skips is not None else None


    @staticmethod
    def _pick_and_prep_skip_from(pool: dict, target: torch.Tensor) -> Optional[torch.Tensor]:

        if not isinstance(pool, dict) or len(pool) == 0:
            return None
        H, W = target.shape[-2:]
        key_exact = f"{H}x{W}"
        if key_exact in pool:
            skip = pool.pop(key_exact)
        else:
            match_key = None
            for k, v in pool.items():
                if hasattr(v, "shape") and v.shape[-2:] == (H, W):
                    match_key = k
                    break
            if match_key is not None:
                skip = pool.pop(match_key)
            else:
                # fallback: take an arbitrary one
                _, skip = pool.popitem()
        skip = skip.to(target.device, dtype=target.dtype)
        if skip.shape[-2:] != (H, W):
            skip = F.interpolate(skip, size=(H, W), mode="bilinear", align_corners=False)
        return skip

    def _fuse_unet_skip_if_needed(
        self,
        h: torch.Tensor,
        prev_hw: Tuple[int, int],
        local_skip_pool: Optional[dict],
        mode: str = "concat",
    ) -> torch.Tensor:
        
        if not isinstance(local_skip_pool, dict) or len(local_skip_pool) == 0:
            return h
        # Trigger only on resolution change (i.e., after an upsample)
        if h.shape[-2:] == prev_hw:
            return h

        if mode != "concat":
            return h  # (future: support 'add', etc.)

        skip = self._pick_and_prep_skip_from(local_skip_pool, h)
        if skip is None:
            return h

        cat = torch.cat([h, skip], dim=1)
        in_ch = cat.shape[1]
        out_ch = h.shape[1]
        H, W  = h.shape[-2:]
        adapter_key = f"cat_{in_ch}->{out_ch}@{H}x{W}"

        if adapter_key not in self.post_concat_adapters:
            # If this ever happens, it means the architecture or skip selection
            # changed in a way that _init_unet_post_concat_adapters didn't anticipate.
            raise RuntimeError(
                f"Missing UNet adapter {adapter_key}. "
                f"You may need to update _init_unet_post_concat_adapters "
                f"for the new architecture / skip shapes."
            )

        adapter = self.post_concat_adapters[adapter_key]

        if self.use_checkpoint and self.training:
            # checkpoint the 1x1 conv on the concatenated feature
            return torch.utils.checkpoint.checkpoint(
                adapter, cat, use_reentrant=False, preserve_rng_state=True
            )
        else:
            return adapter(cat)

    def forward(self, z: torch.Tensor, skips: Optional[dict] = None):
        # Determine skip source - prefer explicitly passed skips for checkpoint safety
        skip_source = skips if skips is not None else self._pending_skips
        
        # Create a local working copy that can be safely mutated
        local_skip_pool = dict(skip_source) if skip_source is not None else {}
        
        # Determine mode - use passed skips' implied mode or fall back to stored mode
        mode = self._unet_mode if skips is None else "concat"
        
        # Track whether we're using skips for this forward pass
        enable_unet = bool(local_skip_pool)

        # 1) Latent projection
        if self.use_checkpoint and self.training:
            h = torch.utils.checkpoint.checkpoint(self.mlp, z, use_reentrant=False, preserve_rng_state=True)
            h =  self.unflatten(h)
        else:
            h = self.unflatten(self.mlp(z))

        # 2) Progressive decoding with optional skip fusion
        for block in self.decoder_blocks:
            prev_hw = h.shape[-2:]
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False, preserve_rng_state=True)
            else:
                h = block(h)
            # fuse skip only if spatial size changed (i.e., just upsampled)
            if enable_unet:
                h = self._fuse_unet_skip_if_needed(h, prev_hw, local_skip_pool, mode)

        # 3) Final head
        if self.use_checkpoint and self.training:
            h = torch.utils.checkpoint.checkpoint(
                lambda x: self.final_act(self.final_norm(x)), h, use_reentrant=False, preserve_rng_state=True
            )
            h_half = torch.utils.checkpoint.checkpoint(self.pre_output, h, use_reentrant=False, preserve_rng_state=True)
            out = torch.utils.checkpoint.checkpoint(self.mdl_head, h_half, use_reentrant=False, preserve_rng_state=True)
        else:
            h = self.final_act(self.final_norm(h))
            out = self.mdl_head(self.pre_output(h))

        return out  # typically (logit_probs, means, log_scales, coeffs)

    def decode(self, z: torch.Tensor, deterministic: bool = False, scale_temp: float = 1.0):
        """Decode latents to images for inference/visualization."""
        logit_probs, means, log_scales, coeffs = self.forward(z)
        if deterministic:
            # Use the mean from the most likely mixture component
            k = logit_probs.argmax(1, keepdim=True)  # [B,1,H,W]
            idx = k.expand(-1, means.size(2), -1, -1)  # [B,C,H,W]
            return means.gather(1, idx)  # [B,C,H,W]
        else:
            # Sample from discretized logistic mixture (simple temperature scaling on logits)
            return self.mdl_head.sample(logit_probs / scale_temp, means, log_scales, coeffs)


#  DINOv3 style


class GramLoss(nn.Module):
    """Implementation of the gram loss"""

    def __init__(
        self,
        apply_norm=True,
        img_level=True,
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()

        # Loss
        self.mse_loss = torch.nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):


        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        # Float casting
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # SSL correlation
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            target_feats = target_feats.flatten(0, 1)

        # Compute similarities
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # Patch correlation
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            output_feats = output_feats.flatten(0, 1)

        # Compute similarities
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim  = torch.where(target_sim  < 0, torch.zeros_like(target_sim),  target_sim)
            student_sim = torch.where(student_sim < 0, torch.zeros_like(student_sim), student_sim)

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            tneg = target_sim < 0
            sneg = student_sim < 0

            # zero where target is negative
            target_sim  = torch.where(tneg, torch.zeros_like(target_sim), target_sim)
            # zero where both were negative
            student_sim = torch.where(sneg & tneg, torch.zeros_like(student_sim), student_sim)

        return self.mse_loss(student_sim, target_sim)