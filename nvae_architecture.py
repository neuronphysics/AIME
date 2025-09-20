import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Optional, Tuple
import logging
import abc
import math



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


# ============= NVAE-style Residual Blocks =============

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


# ============= Main VAE Components =============

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
                layer_idx += 1
        
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
        """
        Enable gradient checkpointing for memory-efficient training.
        This will trade compute for memory by recomputing activations during backward pass.
        """
        self.use_checkpoint = True
    
    def forward(self, x):
        # Stem
        h = self.stem(x)
        
        # Progressive encoding
        skip_connections = []
        for block in self.encoder_blocks:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False, preserve_rng_state=True)
            else:
                h = block(h)
            skip_connections.append(h)
        
        # Final processing
        h = self.final_norm(h)
        h = self.final_act(h)
        
        # MLP
        if self.use_checkpoint and self.training:
            # Checkpoint the MLP computation
            self.hlayer = torch.utils.checkpoint.checkpoint(
                self.mlp, h, use_reentrant=False, preserve_rng_state=True
            )
            # Checkpoint the projection to mean/logvar
            mean_logvar = torch.utils.checkpoint.checkpoint(
                self.proj, self.hlayer, use_reentrant=False, preserve_rng_state=True
            )
        else:
            self.hlayer = self.mlp(h)
            mean_logvar = self.proj(self.hlayer)

        mean, logvar = mean_logvar.chunk(2, dim=-1)
        # Project to latent space
        logvar = torch.clamp(logvar, -10.0, 2.0)  # numerical stability
        sigma = (logvar * 0.5).exp()
        eps= torch.randn_like(sigma)
        z = mean + eps * sigma

        return z, mean, logvar
    #-------------- Fuse Features from different layers ---------------
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
                level += 1
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
        """
        Build a dict of encoder feature maps keyed by spatial size like "HxW".
        Convenience wrapper around extract_pyramid(...) for UNet-style decoding.
        """
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
            feats = self.extract_pyramid(sample_input, levels=source)  # same taps you’ll use at train time
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
            
# ============= Mixture of Discretized Logistics =============
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
        """
        scale_temp (0<τ<=1): shrink logistic scales at sampling time (s' = τ * s).
        mode: "sample" (stochastic) | "mean" (deterministic, per-component mean with RGB coupling)
        """
        B, K, H, W = logit_probs.shape

        # one helper for both branches
        def samp(mu, log_s):
            u = torch.rand_like(mu).clamp_(1e-5, 1 - 1e-5)
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
                return x.clamp_(-1, 1)

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
                return m.clamp_(-1, 1)

            ls = ls + math.log(max(scale_temp, 1e-6))
            x  = samp(m, ls)  # elementwise

        return x.clamp_(-1, 1)
    
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
        dropout: float = 0.0
    ):
        super().__init__()
        self.downsample = downsample
        self.use_checkpoint : bool = False
        # Compute initial spatial dimensions
        initial_h = height // (2 ** downsample)
        initial_w = width // (2 ** downsample)
        initial_channels = channel_size_per_layer[0]
        
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
                    upsample_count += 1
                
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
                layer_idx += 1
        
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
        self._unet_mode: str = "concat"          # currently only "concat" is implemented
        self.post_concat_adapters = nn.ModuleDict()  # 1x1 convs to map cat(C_in+skip) -> C_in

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing.
        """
        self.use_checkpoint = False

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for memory-efficient training.
        This will trade compute for memory by recomputing activations during backward pass.
        """
        self.use_checkpoint = True

    def set_unet_skips(self, skips: Optional[dict], mode: str = "concat"):
        """Provide encoder feature maps for UNet-style fusion. Call right before forward(z).
        Args:
            skips: dict of {"HxW": Tensor[B,C,H,W]} or arbitrary keys -> Tensor[B,C,H,W].
                   If keys aren't 'HxW', we'll match by tensor spatial size.
            mode:  currently only 'concat' is supported.
        """
        
        self._unet_mode = mode
        # Store as a plain dict; do NOT consume/mutate this inside forward (checkpoint-safety).
        self._pending_skips = dict(skips) if skips is not None else None

    # --------------------------------
    # Internal: skip selection (pure)
    # --------------------------------
    @staticmethod
    def _pick_and_prep_skip_from(pool: dict, target: torch.Tensor) -> Optional[torch.Tensor]:
        """Pick a skip from a *local copy* of the pool that best matches target HxW.
        This method is pure w.r.t module state and safe under checkpoint recomputation.
        Mutates only the provided local 'pool' dict.
        """
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
        """Fuse a skip feature when resolution changed (after upsample). Pure wrt module state
        except possibly creating a 1x1 adapter (idempotent across recomputations)."""
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
        # Lazily create adapter once; harmless on recomputation since key already exists.
        if adapter_key not in self.post_concat_adapters:
            self.post_concat_adapters[adapter_key] = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True).to(h.device)
        return self.post_concat_adapters[adapter_key](cat)

    # -------------
    # Forward pass
    # -------------
    def forward(self, z: torch.Tensor, skips: Optional[dict] = None):
        """
        Forward pass with optional skip connections.
        
        Args:
            z: Latent tensor
            skips: Optional dict of skip connections. If not provided, uses self._pending_skips
                   (for backward compatibility with set_unet_skips API)
        
        Returns:
            Output from MDL head (logit_probs, means, log_scales, coeffs)
        """
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
            h = torch.utils.checkpoint.checkpoint(lambda x: self.unflatten(x), h, use_reentrant=False, preserve_rng_state=True)
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

    # ------------------------
    # Convenience for sampling
    # ------------------------
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
            return self.mdl_head.sample_from_mixture(logit_probs / scale_temp, means, log_scales, coeffs)


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
        """Compute the MSE loss between the gram matrix of the input and target features.

        Args:
            output_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            target_feats: Pytorch tensor (B, N, dim) or (B*N, dim) if img_level == False
            img_level: bool, if true gram computed at the image level only else over the entire batch
        Returns:
            loss: scalar
        """

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
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)