import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import List, Optional, Tuple
import logging
import abc
import math


# ============= Utility Functions =============

def swish(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


# ============= Squeeze and Excitation =============

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


# ============= Spectral Normalization =============

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
            blocks.append(nn.BatchNorm2d(out_channels))
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
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
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
                self.mlp, h, use_reentrant=False
            )
            # Checkpoint the projection to mean/logvar
            mean_logvar = torch.utils.checkpoint.checkpoint(
                self.proj, self.hlayer, use_reentrant=False
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
        if out_ch != 3:
            self.params_pre_mix =1+2*out_ch
        else:
            self.params_pre_mix = 10 # 1 logit + 3 means + 3 log_scales + 3 coeffs
        # logits + means(3) + log_scales(3) + coeffs(3) = 10 per mixture
        self.proj = nn.Conv2d(in_ch, n_mix * self.params_pre_mix, 1, bias=True)

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
    def sample(self, logit_probs, means, log_scales, coeffs):
        # Select mixture per pixel
        B, K, H, W = logit_probs.shape
        mix = torch.distributions.Categorical(logits=logit_probs.permute(0,2,3,1)).sample()  # [B,H,W]
        sel = mix.unsqueeze(1).unsqueeze(1)  # [B,1,1,H,W]
        m  = means.gather(1, sel.expand(-1,1,3,-1,-1)).squeeze(1)        # [B,3,H,W]
        ls = log_scales.gather(1, sel.expand(-1,1,3,-1,-1)).squeeze(1)   # [B,3,H,W]
        cs = coeffs.gather(1, sel.expand(-1,1,3,-1,-1)).squeeze(1)       # [B,3,H,W]
        
        # Sample autoregressively with logistic noise
        def samp(mu, log_s):
            u = torch.rand_like(mu).clamp_(1e-5, 1-1e-5)
            return mu + torch.exp(log_s) * (torch.log(u) - torch.log1p(-u))
        
        x_r = samp(m[:,0], ls[:,0])
        x_g = samp(m[:,1] + cs[:,0]*x_r, ls[:,1])
        x_b = samp(m[:,2] + cs[:,1]*x_r + cs[:,2]*x_g, ls[:,2])
        x = torch.stack([x_r, x_g, x_b], dim=1)
        return x.clamp_(-1, 1)
    
    def nll(self, x, logit_probs, means, log_scales, coeffs):
        """
        x: [B,3,H,W] in [-1,1]
        returns per-image NLL (mean over pixels & channels).
        """
        B, _, H, W = x.shape
        K = self.n_mix
        bin_size = 2.0 / 255.0
        
        # Expand x to [B,K,3,H,W]
        xk = x.unsqueeze(1).expand(-1, K, -1, -1, -1)
        
        # Coupled means per channel
        mu_r = means[:, :, 0]
        mu_g = means[:, :, 1] + coeffs[:, :, 0] * xk[:, :, 0]
        mu_b = means[:, :, 2] + coeffs[:, :, 1] * xk[:, :, 0] + coeffs[:, :, 2] * xk[:, :, 1]
        mu = torch.stack([mu_r, mu_g, mu_b], dim=2)  # [B,K,3,H,W]
        inv_s = torch.exp(-log_scales)               # [B,K,3,H,W]
        
        # CDF at bin edges
        x_lo = (xk - bin_size/2) * inv_s + (-mu)*inv_s
        x_hi = (xk + bin_size/2) * inv_s + (-mu)*inv_s
        cdf_lo = torch.sigmoid(x_lo)
        cdf_hi = torch.sigmoid(x_hi)
        
        # For edge bins
        log_cdf_hi = torch.log(cdf_hi.clamp_min(1e-12))
        log_one_minus_cdf_lo = torch.log((1 - cdf_lo).clamp_min(1e-12))
        
        # Probability mass for discretized logistic
        p = torch.where(
            xk < -0.999,                               # near -1
            torch.exp(log_cdf_hi),
            torch.where(
                xk > 0.999,                            # near +1
                torch.exp(log_one_minus_cdf_lo),
                (cdf_hi - cdf_lo).clamp_min(1e-12)
            )
        )
        
        # Sum over channels, then mix with logits
        log_p = torch.log(p.clamp_min(1e-12)).sum(dim=2)           # [B,K,H,W]
        log_mix = log_p + F.log_softmax(logit_probs, dim=1)        # [B,K,H,W]
        log_px = torch.logsumexp(log_mix, dim=1)                   # [B,H,W]
        
        # Return mean NLL per image
        nll = -log_px.mean(dim=(1,2))                              # [B]
        return nll

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
        self.final_norm = nn.BatchNorm2d(current_channels)
        self.final_act = Swish()
        
        self.pre_output = nn.Sequential(
                        nn.Conv2d(current_channels, current_channels // 2, 1, bias=True),
                        nn.SiLU(),
                    )
                    # MDL head
        self.mdl_head = MDLHead(current_channels // 2, out_ch=reconstruction_channels, n_mix=10)
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
    
    def forward(self, z):
        # MLP
        if self.use_checkpoint and self.training:
            # Checkpoint the MLP computation
            h = torch.utils.checkpoint.checkpoint(
                self.mlp, z, use_reentrant=False
            )
            # Checkpoint the unflatten operation (wrap in a lambda for checkpoint)
            h = torch.utils.checkpoint.checkpoint(
                lambda x: self.unflatten(x), h, use_reentrant=False
            )
        else:
            h = self.mlp(z)
            h = self.unflatten(h)
        
        # Progressive decoding
        for block in self.decoder_blocks:
            if self.use_checkpoint and self.training:  
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        # Final processing with checkpointing
        if self.use_checkpoint and self.training:
            h = torch.utils.checkpoint.checkpoint(
                lambda x: self.final_act(self.final_norm(x)), h, use_reentrant=False
            )
            # Checkpoint the pre_output and mdl_head
            out = torch.utils.checkpoint.checkpoint(
                self.pre_output, h, use_reentrant=False
            )
            result = torch.utils.checkpoint.checkpoint(
                self.mdl_head, out, use_reentrant=False
            )
        else:
            h = self.final_norm(h)
            h = self.final_act(h)
            out = self.pre_output(h)
            result = self.mdl_head(out)
        
        return result
    
    def decode(self, z, deterministic=False):
        """Decode latents to images for inference/visualization"""
        logit_probs, means, log_scales, coeffs = self.forward(z)
        if deterministic:
            # Use the mean of the most likely mixture component
            return means.gather(1, logit_probs.argmax(1, keepdim=True).unsqueeze(2).expand(-1, -1, 3, -1, -1)).squeeze(1)
        else:
            return self.mdl_head.sample(logit_probs, means, log_scales, coeffs)

