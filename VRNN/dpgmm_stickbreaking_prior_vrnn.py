import os
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_torch_cache = _project_root / "results" / "pretrained_weights"
os.environ['TORCH_HOME'] = str(_torch_cache)
# Now safe to import torch
import torch
import torchvision
from logging import config
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Categorical, Independent, MixtureSameFamily
from typing import Dict, Tuple, List, Optional, Any
import sys, gc, copy
from einops import rearrange
from contextlib import contextmanager
from itertools import chain
import numpy as np
import math, inspect
from dataclasses import dataclass
from types import SimpleNamespace
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint as ckpt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_networks import EMA, TemporalDiscriminator, AddEpsilon, check_tensor, ImageDiscriminator
from VRNN.RGB import DynamicWeightAverage
from VRNN.VMRNN_D import VMRNNCore
from VRNN.perceiver.position import RoPE
from vdvae.vae import VDVAE
from vdvae.hps import Hyperparams
from vdvae.vae_helpers import mean_from_discretized_mix_logistic, sample_from_discretized_mix_logistic, draw_gaussian_diag_samples, Slot_Slot_Contrastive_Loss
from vdvae.top_dpgmm_prior import ConditionalTopDPGMM, ReplayBufferConditional, TensorDiagComponentPosterior, sample_slots_conditional_frozen, compute_slot_kl_conditional_frozen, sample_slot_vectors_conditional_frozen

@contextmanager
def apply_emas(*emas):
    for e in emas: e.apply_shadow()
    try:
        yield
    finally:
        for e in reversed(emas): e.restore()


################### Main DPGMMVRNN Class ###################

class DPGMMVariationalRecurrentAutoencoder(nn.Module):
    """
    Using Dirichlet Process GMM Prior with Stick-Breaking, Flow-based Warping, and adversarial training for video prediction.
    This architecture incorporates adaptive temporal dynamics, and attention mechanisms
    """
    def __init__(
        self,
        max_components: int,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        sequence_length: int,
        img_disc_layers:int,
        disc_num_heads:int,
        device: torch.device= torch.device('cuda'),
        patch_size: int = 16,
        input_channels: int = 3,  # Number of input channels (e.g., RGB images)
        learning_rate: float = 1e-5,
        grad_clip:float = 10.0,
        prior_alpha: float = 1.0,  # Add these parameters
        prior_beta: float = 10.0,
        dp_alpha: float = 1.0,
        weight_decay: float = 0.00001,
        warmup_epochs: int = 25,
        dropout: float = 0.1,
        use_ctx_checkpoint: bool = True,
        use_dwa: bool = False,
        dwa_temperature: float = 2.0,
        rollout_adv_every: int = 1,            # do rollout adversarial every N steps (0 disables)
        rollout_context_frames: int = 3,        # T_ctx
        rollout_horizon: int = 4,               # rollout length
        lambda_rollout_adv: float = 1.0,       # strength of rollout adversarial losses
        rollout_top_temperature: float = 1.0,   # sampling temperature for top prior
        patch_disc_layers: int = 2,
        patch_disc_ndf:int = 36,
        mamba_d_state:int =32,
        lecam_ema_decay: float = 0.99,    # EMA decay for LeCam regularization anchors
        top_slot_iters: int = 3,
        slot_diversity_weight: float = 0.5,
        top_slot_dim: Optional[int] = None,
        num_top_slots: Optional[int] = None,
        unfreeze_structural_warmup_epochs: int = 9,
        lambda_slot_contrast: float = 0.3,
        lambda_slot_orth: float = 0.25,
        lambda_mask_entropy: float = 0.01,
        lambda_mask_balance: float = 0.1,
        slot_contrast_temperature: float = 0.1,
        slot_contrast_batch: bool = True,
    ):
        super().__init__()
        # core dimensions
        self.input_channels = input_channels
        self.image_size = input_dim
        self.max_K = max_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.device = device
        self.dropout = dropout
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.dp_alpha = float(dp_alpha)
        self.init_components = min(5, int(max_components))
        # Hyperparameters
        self._lr = learning_rate
        self._grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.use_dwa = use_dwa
        self.use_ctx_checkpoint = use_ctx_checkpoint
        self.patch_disc_layers=patch_disc_layers
        self.patch_disc_ndf = patch_disc_ndf
        self.eps = torch.finfo(torch.float32).eps
        # KL overshooting
        self.overshoot_w_decay = 0.9
        self.lambda_overshoot = 0.5
        # rollout GAN attributes
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.mamba_d_state = int(mamba_d_state)
        self.rollout_adv_every = int(rollout_adv_every)
        self.rollout_context_frames = int(rollout_context_frames)
        self.rollout_horizon = int(rollout_horizon)
        self.lambda_rollout_adv = float(lambda_rollout_adv)
        self.rollout_top_temperature = float(rollout_top_temperature)

        self.lecam_ema_decay = float(lecam_ema_decay)
        self.num_top_slots = int(num_top_slots) if num_top_slots is not None else self.init_components
        self.slot_diversity_weight = float(slot_diversity_weight)
        self.top_slot_dim = int(top_slot_dim) if top_slot_dim is not None else int(latent_dim * 2)
        self.top_slot_iters = int(top_slot_iters)
        self.unfreeze_structural_warmup_epochs = unfreeze_structural_warmup_epochs
        self.lambda_slot_contrast = float(lambda_slot_contrast)
        self.lambda_slot_orth = float(lambda_slot_orth)
        self.lambda_mask_entropy = lambda_mask_entropy
        self.lambda_mask_balance = lambda_mask_balance
        self.slot_contrast_loss_fn = Slot_Slot_Contrastive_Loss(
            pred_key="top_slot_mu",
            target_key="top_slot_mu",
            temperature=slot_contrast_temperature,
            batch_contrast=slot_contrast_batch,
        )
        # scalar EMA anchors for LeCam regularization
        self.register_buffer("lecam_initialized", torch.tensor(False), persistent=True)

        self.register_buffer("lecam_temporal_real_ema", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer("lecam_temporal_fake_ema", torch.zeros((), dtype=torch.float32), persistent=True)

        self.register_buffer("lecam_patch_real_ema", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer("lecam_patch_fake_ema", torch.zeros((), dtype=torch.float32), persistent=True)

        # initialization different parts of the model
        self._init_encoder_decoder(max_components)

        # default: allow up to one full top-grid width/height in pixels
        self._init_vrnn_dynamics( extra_channels=0)
        self._init_discriminators(img_disc_layers, patch_size, num_heads=disc_num_heads)


        if use_dwa:
            self._init_DynamicWeightAverage(dwa_temperature)

        self.to(device)

        # Setup optimizers
        self._setup_optimizers(learning_rate, weight_decay)

    @torch.no_grad()
    def _update_lecam_ema(
        self,
        temporal_real_mean: torch.Tensor,
        temporal_fake_mean: torch.Tensor,
        patch_real_mean: torch.Tensor,
        patch_fake_mean: torch.Tensor,
    ) -> None:
        tr = temporal_real_mean.detach().to(
            device=self.lecam_temporal_real_ema.device,
            dtype=self.lecam_temporal_real_ema.dtype,
        )
        tf = temporal_fake_mean.detach().to(
            device=self.lecam_temporal_fake_ema.device,
            dtype=self.lecam_temporal_fake_ema.dtype,
        )
        pr = patch_real_mean.detach().to(
            device=self.lecam_patch_real_ema.device,
            dtype=self.lecam_patch_real_ema.dtype,
        )
        pf = patch_fake_mean.detach().to(
            device=self.lecam_patch_fake_ema.device,
            dtype=self.lecam_patch_fake_ema.dtype,
        )

        if not bool(self.lecam_initialized):
            self.lecam_temporal_real_ema.copy_(tr)
            self.lecam_temporal_fake_ema.copy_(tf)
            self.lecam_patch_real_ema.copy_(pr)
            self.lecam_patch_fake_ema.copy_(pf)
            self.lecam_initialized.fill_(True)
            return

        d = self.lecam_ema_decay

        self.lecam_temporal_real_ema.mul_(d).add_(tr, alpha=1.0 - d)
        self.lecam_temporal_fake_ema.mul_(d).add_(tf, alpha=1.0 - d)
        self.lecam_patch_real_ema.mul_(d).add_(pr, alpha=1.0 - d)
        self.lecam_patch_fake_ema.mul_(d).add_(pf, alpha=1.0 - d)

    def _init_DynamicWeightAverage(self, temperature: float = 2.0):
        self.total_weighter = DynamicWeightAverage(
            loss_keys_to_consider=[
                "recon_loss",
                "kl_z",
                "slot_contrast_loss",
                "img_adv_loss",
                "temporal_adv_loss",
                "feat_match_loss",
                "rollout_img_adv_loss",
                "rollout_temporal_adv_loss",
                "rollout_feat_match_loss",
                "overshoot_kl",
            ],
            temperature=temperature,
        )

    @staticmethod
    def to01(x: torch.Tensor) -> torch.Tensor:
        # Commonly observations are in [-1, 1]
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)


    @staticmethod
    def _length_sq(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * x, dim=1, keepdim=True)


    def _maybe_ckpt(self, fn, *args):
        if (
            self.use_ctx_checkpoint
            and self.training
            and any(torch.is_tensor(arg) and arg.requires_grad for arg in args)
        ):
            return ckpt(fn, *args, use_reentrant=False)
        return fn(*args)


    def _init_encoder_decoder(self, max_components: int, prior_mc_samples: int = 35):
        """
        Initialize VDVAE + DPGMM prior.

        """
        # 1) Build VDVAE hyperparams
        H = Hyperparams()
        H.action_dim = self.action_dim      # required by the ConvLSTM temporal prior

        H.use_checkpoint = self.use_ctx_checkpoint
        H.image_channels = self.input_channels   # usually 3
        H.zdim = self.latent_dim                 # or set explicitly (e.g., 16)
        H.bottleneck_multiple = 0.25
        H.width = 32
        H.image_size = self.image_size          # e.g. 64
        H.dataset = 'imagenet64'
        H.num_mixtures = 10
        H.skip_threshold = 100.0
        H.enc_blocks = "64x2,64d2,32x2"
        H.dec_blocks = "32x2,64m32,64x2"
        H.attn_resolutions = []
        H.use_spatial_attn = False
        H.attn_where = "last"
        H.top_h_context_dim = H.zdim + self.action_dim

        H.no_bias_above = 64
        H.custom_width_str = ""
        # --- Attention defaults ---
        H.attn_num_layers = 1
        H.attn_num_heads = 8
        H.attn_widening_factor = 1
        H.attn_dropout = 0.0
        H.attn_residual_dropout = 0.0
        H.attn_gn_groups = 32
        H.attn_pos_num_bands = 6
        H.overshoot_K = 6
        H.top_num_slots = int(self.num_top_slots)
        H.top_slot_dim = self.top_slot_dim
        H.top_slot_iters = self.top_slot_iters
        H.slot_diversity_weight = self.slot_diversity_weight
        H.top_slot_decoder_hidden =  H.width

        # ---- 2) Instantiate VDVAE ----
        self.vdvae = VDVAE(
            H,
            prior=None,              # we'll set self.prior separately
            top_kl_weight=1.0,
            prior_kl_mc_samples=prior_mc_samples,
        ).to(self.device)
        # ---- 3) Extract top block latent dim & resolution ----
        top_block = self.vdvae.decoder.dec_blocks[0]        # is_top=True for first block
        C = top_block.zdim                                  # latent channels
        res = top_block.base
        self.zdim = C                           # spatial resolution (e.g. 8)
        self.top_zdim = C * res * res

        self.top_H = res
        self.top_W = res
        self.top_num_slots = int(H.top_num_slots)
        self.top_slot_dim = int(H.top_slot_dim)
        self.top_prior_model = ConditionalTopDPGMM(
            z_shape=(self.top_slot_dim, 1, 1),
            h_shape=(self.latent_dim + self.action_dim, self.top_H, self.top_W),
            max_components=max_components,
            init_components = self.init_components,
            num_slots=self.top_num_slots,
            dp_alpha = self.dp_alpha,
            prior_alpha0 = self.prior_alpha,
            prior_beta0 = self.prior_beta,
            gate_hidden_dim = self.hidden_dim,
            gate_lr = self._lr * 0.1,
            birth_kfresh = 10,
            birth_resp_threshold = 0.1,
            birth_subset_max= 8192,
            device=self.device,
            dtype=torch.float32,
            use_checkpoint=self.use_ctx_checkpoint,
        )
        self.vdvae.top_prior_ready = False
        self.top_replay_buffer = ReplayBufferConditional()
        # EMA for VDVAE
        self.ema_decay = 0.999
        self.ema_vdvae = EMA(self.vdvae, decay=self.ema_decay)

        # Attach prior to VDVAE so its forward() computes dp_kl / dp_rate
        self.vdvae.top_prior = self.top_prior_model
        self._bootstrap_top_prior_state()
        self.top_prior_initialized_from_data = False
        self.overshoot_K = int(getattr(H, "overshoot_K", 6))


    @torch.no_grad()
    def _bootstrap_top_prior_state(self):
        prior = self.top_prior_model
        prior.K = self.init_components
        prior.gate.set_active_K(prior.K)

        K = prior.K
        device = prior.device_
        dtype = prior.dtype

        m0 = prior._prior_m0()
        beta0 = prior._prior_beta0()

        alpha = torch.tensor(
            prior.prior_alpha0 + 0.5,
            device=device,
            dtype=dtype,
        )
        kappa = torch.tensor(
            prior.prior_kappa0 + 1.0,
            device=device,
            dtype=dtype,
        )

        slot_block = self.vdvae.decoder.dec_blocks[0].top_slot_posterior
        anchors = slot_block.slots_init_mu.detach().to(device=device, dtype=dtype)

        anchors = anchors[:K]  # [K, slot_dim]

        # map through to_mu, because DPGMM is over slot_mu, not raw init slots.
        anchors = slot_block.to_mu(anchors)

        # Center and normalize anchors so they do not become too extreme.
        anchors = anchors - anchors.mean(dim=0, keepdim=True)
        anchors = F.normalize(anchors, dim=-1)

        comp_std = torch.sqrt((beta0 / (alpha - 1.0).clamp_min(1e-3)).clamp_min(1e-6))
        radius = math.sqrt(2.0 * 1.5)

        prior.comp = []
        for k in range(K):
            direction = anchors[k].view_as(m0)
            mean_k = m0 + radius * comp_std * direction

            prior.comp.append(
                TensorDiagComponentPosterior(
                    mean=mean_k.clone(),
                    kappa=kappa.clone(),
                    alpha=alpha.clone(),
                    beta=beta0.clone(),
                )
            )

        prior._reset_global_summaries()

        self.vdvae.top_prior_snapshot = prior.frozen_snapshot()
        self.vdvae.top_prior_gate = ConditionalTopDPGMM.load_gate_from_snapshot(
            snapshot=self.vdvae.top_prior_snapshot,
            h_shape=(self.latent_dim + self.action_dim, self.top_H, self.top_W),
            hidden_dim=prior.gate.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )

    def _init_decoder_prev_latents(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Per decoder block index.
        Top block gets None because it does not use latent_prev.
        Non-top blocks get zero tensors [B, zdim, res, res].
        """
        prev_latents = []
        for idx, block in enumerate(self.vdvae.decoder.dec_blocks):
            if idx == 0:
                prev_latents.append(None)
            else:
                prev_latents.append(
                    torch.zeros(
                        batch_size,
                        block.zdim,
                        block.base,
                        block.base,
                        device=device,
                        dtype=dtype,
                    )
                )
        return prev_latents


    def _init_discriminators(self, img_disc_layers:int, patch_size: int, num_heads: int = 4):
        # Initialize discriminators
        self.image_discriminator = TemporalDiscriminator(
            input_channels=self.input_channels,
            cond_channels=self.zdim+self.latent_dim + self.action_dim,
            num_layers=int(img_disc_layers),
            conv_type="standard",
            attn_dim=self.hidden_dim,
            num_heads=num_heads,
            internal_chn = int(self.hidden_dim),
            use_checkpoint= False,
            ckpt_use_reentrant=False,
        ).to(self.device)
        self.patch_discriminator = ImageDiscriminator(
                input_nc=self.input_channels,
                ndf=int(self.patch_disc_ndf),
                n_layers=int(self.patch_disc_layers),
                norm_type= "frn",
                gn_groups= 8,
                use_checkpoint=False,
                checkpoint_use_reentrant=False,
                device=self.device,
            )


    def _init_vrnn_dynamics(self, extra_channels: int =0):
        """Initialize VRNN components with context conditioning"""
        # Feature extractors
        # VRNN recurrence: h_t = f(h_{t-1}, z_t, a_t)
        self._rnn = VMRNNCore(
            zdim=self.zdim,
            action_dim=self.action_dim,
            height=self.top_H,
            width=self.top_W,
            depth=1,
            d_state=self.mamba_d_state,
            use_checkpoint=self.use_ctx_checkpoint,
        )

    @property
    def rnn(self):
        """Property to access the RNN layer."""
        return self._rnn

    def _setup_optimizers(self, learning_rate: float, weight_decay: float) -> None:
        def collect_unique_params(*items, exclude_params=None):
            params = []
            seen_ids = set()
            exclude_ids = {id(p) for p in (exclude_params or []) if isinstance(p, nn.Parameter)}

            def visit(x):
                if x is None:
                    return

                if isinstance(x, nn.Parameter):
                    if x.requires_grad and id(x) not in exclude_ids and id(x) not in seen_ids:
                        params.append(x)
                        seen_ids.add(id(x))
                    return

                if isinstance(x, nn.Module):
                    for p in x.parameters():
                        if p.requires_grad and id(p) not in exclude_ids and id(p) not in seen_ids:
                            params.append(p)
                            seen_ids.add(id(p))
                    return

                if isinstance(x, (list, tuple, set)):
                    for y in x:
                        visit(y)
                    return

                raise TypeError(f"Unsupported param container type: {type(x)}")

            for item in items:
                visit(item)

            return params

        def split_by_weight_decay(params, wd):
            decay, no_decay = [], []
            for p in params:
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:
                    decay.append(p)
                else:
                    no_decay.append(p)

            groups = []
            if decay:
                groups.append({"params": decay, "weight_decay": wd})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            return groups

        def add_groups(dst, params, lr, wd):
            if not params:
                return
            for g in split_by_weight_decay(params, wd):
                g["lr"] = lr
                dst.append(g)

        # LR / Weight Decay policy
        base_lr = float(learning_rate)

        lr_core  = base_lr * 1.0
        lr_state = base_lr * 1.0

        wd_core  = float(weight_decay)

        # Special params first
        scalar_state_params = [
            getattr(self.rnn, "h0", None),
            getattr(self.rnn, "c0", None),
        ]
        scalar_state_params = [p for p in scalar_state_params if isinstance(p, nn.Parameter) and p.requires_grad]

        top_prior_params = [p for p in self.top_prior_model.parameters() if p.requires_grad]
        special_exclude = scalar_state_params + top_prior_params

        core_params = collect_unique_params(
            self.vdvae,
            self.rnn,
            exclude_params=special_exclude,
        )
        gen_param_groups = []
        add_groups(gen_param_groups, core_params,  lr_core,  wd_core)

        if scalar_state_params:
            gen_param_groups.append({
                "params": scalar_state_params,
                "lr": lr_state,
                "weight_decay": 0.0,
            })

        self.gen_optimizer = torch.optim.AdamW(
            gen_param_groups,
            lr=lr_core,
            betas=(0.9, 0.999),
            eps=1e-5,
        )
        # Discriminator optimizer
        disc_param_groups = []

        if hasattr(self, "image_discriminator") and self.image_discriminator is not None:
            disc_param_groups.append({
                "params": [p for p in self.image_discriminator.parameters() if p.requires_grad],
                "lr": base_lr * 0.5,
            })

        if hasattr(self, "patch_discriminator") and self.patch_discriminator is not None:
            disc_param_groups.append({
                "params": [p for p in self.patch_discriminator.parameters() if p.requires_grad],
                "lr": base_lr * 0.07,
            })

        # flatten away any empty groups
        disc_param_groups = [g for g in disc_param_groups if len(g["params"]) > 0]

        self.img_disc_optimizer = None
        if disc_param_groups:
            self.img_disc_optimizer = torch.optim.Adamax(
                disc_param_groups,
                betas=(0.0, 0.9),
                weight_decay=5e-5,
            )

        gen_ids = [id(p) for g in self.gen_optimizer.param_groups for p in g["params"]]
        if len(gen_ids) != len(set(gen_ids)):
            raise RuntimeError("Duplicate parameter detected inside gen_optimizer.")

        disc_ids = []
        if self.img_disc_optimizer is not None:
            disc_ids = [id(p) for g in self.img_disc_optimizer.param_groups for p in g["params"]]
            if len(disc_ids) != len(set(disc_ids)):
                raise RuntimeError("Duplicate parameter detected inside img_disc_optimizer.")

        overlap = set(gen_ids).intersection(disc_ids)
        if overlap:
            raise RuntimeError("Some parameters are present in both generator and discriminator optimizers.")

        # expected generator params = all trainable params in self minus discriminator params
        excluded_from_gen = {
            id(p) for p in self.top_prior_model.parameters()
            if p.requires_grad
        }
        all_trainable_ids = {
            id(p) for p in self.parameters()
            if p.requires_grad and id(p) not in excluded_from_gen
        }
        expected_disc_ids = set(disc_ids)
        expected_gen_ids = all_trainable_ids - expected_disc_ids

        if set(gen_ids) != expected_gen_ids:
            missing = expected_gen_ids - set(gen_ids)
            extra = set(gen_ids) - expected_gen_ids
            raise RuntimeError(
                f"Generator optimizer coverage mismatch: missing={len(missing)}, extra={len(extra)}"
            )

        self._setup_schedulers()


    def _setup_schedulers(self):
        """Setup learning-rate schedulers for all optimizers (Option B)."""
        # 1) Trunk scheduler (PCGrad or not, this uses the base optimizer)
        self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gen_optimizer,
            mode='min',
            factor=0.2,
            patience=10,
            threshold=0.0001,
        )
        # 3) Discriminator scheduler (unchanged)
        if hasattr(self, "img_disc_optimizer"):
            self.img_disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.img_disc_optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                min_lr=1e-7,
            )

    def refresh_top_prior_from_buffer(self, batch_size: int =256, n_laps: int =4):
        if self.top_replay_buffer.is_empty():
            return
        structural_allowed = self.current_epoch >= self.unfreeze_structural_warmup_epochs
        self.vdvae.top_prior_snapshot = self.top_prior_model.fit_epoch(
            buffer=self.top_replay_buffer,
            batch_size=batch_size,
            init_K=self.top_prior_model.K,
            n_laps=n_laps,
            do_birth=structural_allowed,
            do_merge=structural_allowed,
            do_delete=structural_allowed,
            warm_start=self.top_prior_initialized_from_data,
        )

        self.top_prior_initialized_from_data = True
        with torch.no_grad():
            self.vdvae.top_prior_gate = ConditionalTopDPGMM.load_gate_from_snapshot(
                self.vdvae.top_prior_snapshot,
                h_shape=(self.latent_dim + self.action_dim, self.top_H, self.top_W),
                hidden_dim=self.top_prior_model.gate.hidden_dim,
                device=self.device,
                dtype=torch.float32,
            )
        self.vdvae.top_prior_ready = True
        self.top_replay_buffer.clear()
        self.top_prior_model.clear_fit_cache()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    def _lengths_from_dones(self, dones: torch.Tensor, T: int, assume_padded_after_done: bool = True):
        """
        dones: [B, T] where dones[t]=1 means episode terminates AFTER frame t (i.e., t is valid, t+1 invalid).
        returns:
        alive_mask: [B, T] bool (True = valid frame)
        lengths:    [B] long (#valid frames)
        """
        B = dones.shape[0]
        dones = dones[:, :T].bool()

        alive = torch.ones(B, T, device=dones.device, dtype=torch.bool)
        if assume_padded_after_done:
            # once done happens at time t, frames t+1, t+2, ... are invalid
            if T > 1:
                done_prev = dones[:, :T-1]                        # termination on transition to next frame
                ended_before = (done_prev.cumsum(dim=1) > 0)      # [B, T-1]
                alive[:, 1:] = ~ended_before

        lengths = alive.long().sum(dim=1)
        return alive, lengths


    def forward_sequence(
        self,
        observations,
        actions=None,
        dones=None,
        store_reconstruction_samples: bool = False,
        collect_top_buffer: bool = False,
        seq_ids: Optional[torch.Tensor] = None,
    ):
        """
        observations: [B, T, C, H, W] in [-1, 1]
        actions:      [B, T, action_dim]
        dones:        [B, T] optional; expects done[t-1] to reset state at step t

        Returns a dict of per-timestep outputs.
        """
        # Unpack basic shapes and default action tensor
        batch_size, seq_len, C, H, W = observations.shape
        device = observations.device
        dtype = observations.dtype
        B = batch_size

        if actions is None:
            actions = torch.zeros(batch_size, seq_len, self.action_dim, device=device, dtype=dtype)
        batch_seq_ids = None
        if collect_top_buffer:
            if seq_ids is not None:
                batch_seq_ids = seq_ids.to(device=device, dtype=torch.long).reshape(B)
            else:
                batch_seq_ids = self.top_replay_buffer.allocate_seq_ids(B, device=device)
        # Initialize recurrent state and top-grid context
        core_state = self.rnn.init_state(batch_size, device=device, dtype=dtype)

        z_prev_top = torch.zeros(B, self.zdim, self.top_H, self.top_W, device=device, dtype=dtype)
        prev_latents = self._init_decoder_prev_latents(B, device, dtype)

        # Prepare output containers
        outputs = {
            "reconstructions": [],
            "reconstruction_samples": [] if store_reconstruction_samples else None,
            "latents": [],
            "hidden_states": [],              # post-RNN-update normalized hidden maps, flattened
            "core_h_maps": [],
            "core_c_maps": [],
            "prior_pi": [],
            "reconstruction_losses": [],
            "gauss_rate": [],
            "dp_rate": [],
            "kl_latents": [],
            "elbo": [],
            "top_q_mean_map": [],
            "top_q_logvar_map": [],
            "z_seq_maps": [],
            "top_slot_mu": [],
            "top_slot_logsigma": [],
            "top_slot_resp": [],
            "top_slot_attn": [],
            "decoder_masks": [],
            "top_bridge_rate": [],
            # overshoot anchor for the new transport dynamics
            "overshoot_z_prev_top": [],
        }
        # Initialize previous top-posterior mean for LatentWarp
        # NOTE: channel dim must match the top latent map, i.e. self.zdim

        for t in range(seq_len):
            # Read the current observation x_t
            x_t = observations[:, t]
            #if t == 0: print(f"observation range: [{x_t.min().item():.4f}, {x_t.max().item():.4f}]")

            # Use previous action a_{t-1} for the current recurrent step
            a_t = (
                torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype)
                if t == 0
                else actions[:, t - 1]
            )

            # Build keep-mask for step t; resets state if done at t-1
            if dones is None or t == 0:
                mask_t = torch.ones(batch_size, device=device, dtype=torch.float32)
            else:
                mask_t = (1.0 - dones[:, t - 1].float()).to(torch.float32)

            h, c = core_state
            keep = mask_t.view(B, 1, 1).to(device=device, dtype=dtype)

            h0, c0= self.rnn.init_state(B, device=device, dtype=dtype)
            core_state = (
                h * keep + h0 * (1.0 - keep),
                c * keep + c0 * (1.0 - keep),
            )
            z_prev_top_masked = z_prev_top * mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype)
            assert all(pl is None or pl.shape[0] == B or pl.shape[0] % B == 0 for pl in prev_latents)
            prev_latents = [None if pl is None else pl * (mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype) if pl.shape[0] == B else mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype).repeat_interleave(pl.shape[0] // B, dim=0)) for pl in prev_latents]
            h_tok = self.rnn.out_norm(core_state[0])
            B_, _, D_ = h_tok.shape
            h_t= h_tok.transpose(1, 2).reshape(B_, D_, self.top_H, self.top_W ).contiguous()


            # Encode the current frame with VDVAE conditioned on current context
            x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
            vdvae_out = self.vdvae.forward(
                x_t_nhwc,
                x_t_nhwc,
                h_prior_top=h_t,
                mask_t=mask_t,
                prev_latents=prev_latents,
                get_latents=False,
            )

            z_top_state = vdvae_out.get("z_top_state", vdvae_out["current_latents"][0])
            B2, zdim, Ht, Wt = z_top_state.shape
            t_index = torch.full((B,), t, device=device, dtype=torch.int64)
            # collect posterior samples for epoch-end DPGMM fitting
            if collect_top_buffer:
                seq_id_t = batch_seq_ids.to(torch.int64)
                top_slot_z= vdvae_out["top_slot_mu"] + torch.exp(vdvae_out["top_slot_logsigma"]) * torch.randn_like(vdvae_out["top_slot_mu"])
                Bs, S, D =top_slot_z.shape

                self.top_replay_buffer.add_step_batch(
                    h_t=h_t.detach(),                                      # [B,C,H,W], exact context, not expanded
                    z_top_map=top_slot_z.detach(),            # [B,S,D]
                    posterior_mean=vdvae_out["top_slot_mu"].detach(),      # [B,S,D]
                    posterior_logvar=(2.0 * vdvae_out["top_slot_logsigma"]).detach(),  # [B,S,D]
                    valid_mask=mask_t.bool(),                              # [B]
                    seq_id=seq_id_t,                                       # [B]
                    t_index=t_index,                                       # [B]
                )
            elogpi_t = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
                self.vdvae.top_prior_gate,
                h_t,
            )
            pi_t = torch.softmax(elogpi_t, dim=1).detach()
            # Flatten latent for discriminator conditioning

            a_cur = actions[:, t]
            # Update recurrent state using current z_t, previous action a_{t-1}, and motion-dependent extra maps
            h_context_next, core_state = self.rnn(
                z_top_state,
                a_cur,
                state=core_state,
                mask_t=None,
                extra_maps=None,
            )

            z_prev_top = z_top_state.detach()
            prev_latents = [None if idx == 0 else pl.detach() for idx, pl in enumerate(vdvae_out["current_latents"])]

            # Store recurrent states and latent statistics for this timestep
            outputs["core_h_maps"].append(core_state[0])
            outputs["core_c_maps"].append(core_state[1])
            outputs["prior_pi"].append(pi_t)
            outputs["top_q_mean_map"].append(vdvae_out["top_q_mean_map"].detach())
            outputs["top_q_logvar_map"].append(vdvae_out["top_q_logvar_map"].detach())
            outputs["overshoot_z_prev_top"].append(z_prev_top)
            ##slot component outputs (for analysis / visualization)
            top_slot_mu = vdvae_out.get("top_slot_mu", None)
            top_slot_logsigma = vdvae_out.get("top_slot_logsigma", None)
            top_slot_resp = vdvae_out.get("top_slot_resp", None)
            top_slot_attn = vdvae_out.get("top_slot_attn", None)

            if top_slot_mu is not None:
                outputs["top_slot_mu"].append(top_slot_mu)

            if top_slot_logsigma is not None:
                outputs["top_slot_logsigma"].append(top_slot_logsigma)

            # top_slot_resp only exists after the top prior is ready.
            # During epoch 0 / warmup, use the current prior weights as a harmless placeholder.
            if top_slot_resp is None and top_slot_mu is not None:
                Bs, S, _ = top_slot_mu.shape
                top_slot_resp = pi_t[:, None, :].expand(Bs, S, -1).contiguous()

            if top_slot_resp is not None:
                outputs["top_slot_resp"].append(top_slot_resp.detach())

            if top_slot_attn is not None:
                outputs["top_slot_attn"].append(top_slot_attn.detach())



            # Decode reconstruction mean and log scalar VAE losses
            px_z = vdvae_out["px_z"]                         # [B*S,width+1,H,W]
            dmol_out = self.vdvae.decoder.out_net.forward(px_z)

            rgb_slots = mean_from_discretized_mix_logistic(
                dmol_out,
                self.vdvae.H.num_mixtures,
            )  # [B*S,H,W,3]

            N, H_img, W_img, _ = rgb_slots.shape

            S = N // B

            rgb_slots = rgb_slots.view(B, S, H_img, W_img, 3).permute(0, 1, 4, 2, 3)
            # [B,S,3,H,W]

            mask_logits = px_z[:, self.vdvae.H.width:self.vdvae.H.width + 1]
            mask_logits = mask_logits.view(B, S, 1, H_img, W_img)

            decoder_masks = F.softmax(mask_logits, dim=1)
            outputs["decoder_masks"].append(decoder_masks)
            # [B,S,1,H,W]

            recon_mean = torch.sum(decoder_masks * rgb_slots, dim=1)
            # [B,3,H,W]

            if store_reconstruction_samples:
                with torch.no_grad():
                    sample = self.vdvae.decoder.out_net.sample(vdvae_out["px_z"])
                    sample = torch.from_numpy(sample).to(device=device, dtype=torch.float32)
                    sample = sample.permute(0, 3, 1, 2).contiguous()/ 127.5 - 1.0 #TODO:is this correct

                    sample_img = sample.clamp(-1.0, 1.0)
                outputs["reconstruction_samples"].append(sample_img)
            outputs["reconstructions"].append(recon_mean)
            outputs["latents"].append(z_top_state.detach())
            ctx_dim = h_context_next.shape[1]
            outputs["hidden_states"].append( h_context_next.permute(0, 2, 3, 1).contiguous().view(batch_size, Ht * Wt * ctx_dim).detach())
            outputs["z_seq_maps"].append(torch.cat([z_top_state.detach(), h_context_next.detach()], dim=1))
            outputs["reconstruction_losses"].append(vdvae_out["distortion"])
            outputs["gauss_rate"].append(vdvae_out["gauss_rate"].detach())
            outputs["dp_rate"].append(vdvae_out["dp_rate"].detach())
            outputs["kl_latents"].append(vdvae_out["rate"])
            outputs["elbo"].append(vdvae_out["elbo"].detach())
            outputs["top_bridge_rate"].append(vdvae_out["top_bridge_rate"].detach())

        # Stack time-major tensors where downstream code expects [B, T, ...]
        for k in [
            "core_h_maps",
            "core_c_maps",
            "overshoot_z_prev_top",
            "z_seq_maps",
            "top_slot_mu",
            "top_slot_logsigma",
            "top_slot_resp",
            "top_slot_attn",
            "decoder_masks",
        ]:
            if len(outputs[k]) > 0:
                outputs[k] = torch.stack(outputs[k], dim=1)
        if len(outputs["top_bridge_rate"]) > 0:
            outputs["top_bridge_rate_seq"] = torch.stack(
                [x.reshape(()) for x in outputs["top_bridge_rate"]],
                dim=0,
            )  # [T]
            outputs["top_bridge_rate"] = outputs["top_bridge_rate_seq"].mean()

        for key in ["reconstructions", "latents", "hidden_states"]:
            if len(outputs[key]) > 0 and isinstance(outputs[key][0], torch.Tensor):
                outputs[key] = torch.stack(outputs[key], dim=1)

        if store_reconstruction_samples and len(outputs["reconstruction_samples"]) > 0:
            outputs["reconstruction_samples"] = torch.stack(outputs["reconstruction_samples"], dim=1)

        outputs["top_q_mean_maps"] = torch.stack(outputs.pop("top_q_mean_map"), dim=1)
        outputs["top_q_logvar_maps"] = torch.stack(outputs.pop("top_q_logvar_map"), dim=1)

        return outputs


    def compute_lecam_loss(
        self,
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
        ema_logits_real: torch.Tensor,
        ema_logits_fake: torch.Tensor
    ) -> torch.Tensor:
        """Computes the LeCam loss for the given average real and fake logits.

        Returns:
            lecam_loss -> torch.Tensor: The LeCam loss.
        """
        lecam_loss = torch.pow(F.relu(logits_real - ema_logits_fake), 2).mean()
        lecam_loss += torch.pow(F.relu(ema_logits_real - logits_fake), 2).mean()
        return lecam_loss

    def compute_multistep_kl_overshoot(
        self,
        top_slot_mean: torch.Tensor,        # [B,T,S,D]
        top_slot_logsigma: torch.Tensor,    # [B,T,S,D]
        actions,              # [B,T,A]
        dones: Optional[torch.Tensor] = None,
        K:int=15,
        w_decay=0.9,
        core_state_maps=None,          # (h,c,m): [B,T,hidden,Ht,Wt]
        overshoot_anchor_state=None,   # dict
        rollout_temperature: float = 1.0,
    ):
        device = top_slot_mean.device
        dtype = top_slot_mean.dtype
        B, T, S, D = top_slot_mean.shape
        if (
            not bool(getattr(self.vdvae, "top_prior_ready", False))
            or self.vdvae.top_prior_snapshot is None
            or self.vdvae.top_prior_gate is None
        ):
            return torch.zeros((), device=device, dtype=dtype)

        if T < 2 or K <= 0:
            return torch.zeros((), device=device, dtype=dtype)

        if dones is not None:
            alive, _ = self._lengths_from_dones(dones, T, assume_padded_after_done=True)
        else:
            alive = torch.ones(B, T, device=device, dtype=torch.bool)

        if core_state_maps is None or overshoot_anchor_state is None:
            raise ValueError("Pass core_state_maps and overshoot_anchor_state.")

        core_h_seq, core_c_seq= core_state_maps
        z_prev_seq = overshoot_anchor_state["z_prev_top"]

        A = T - 1
        anchors = torch.arange(A, device=device) #0, ..., T-2
        BA = B * A

        anchor_alive = alive[:, anchors]  # [B,A]

        h = core_h_seq[:, anchors].reshape(BA, *core_h_seq.shape[2:]).contiguous().detach()
        c = core_c_seq[:, anchors].reshape(BA, *core_c_seq.shape[2:]).contiguous().detach()
        z_prev_top = z_prev_seq[:, anchors].reshape(BA, *z_prev_seq.shape[2:]).contiguous().detach()


        topH, topW = self.top_H, self.top_W

        numer = torch.zeros((), device=device, dtype=dtype)
        denom = torch.zeros((), device=device, dtype=dtype)

        for k in range(1, K + 1):
            t_idx = anchors + k
            in_range = (t_idx < T)
            if not in_range.any():
                break

            t_idx_safe = t_idx.clamp(max=T - 1)

            future_alive = torch.zeros(B, A, device=device, dtype=torch.bool)
            future_alive[:, in_range] = alive[:, t_idx[in_range]]

            valid_ba = (anchor_alive & future_alive).reshape(BA)
            if not valid_ba.any():
                continue

            mask_roll = valid_ba.to(dtype)

            keep = mask_roll.view(BA, 1, 1).to(device=device, dtype=dtype)
            h0, c0= self.rnn.init_state(BA, device=device, dtype=dtype)
            h = h * keep + h0 * (1.0 - keep)
            c = c * keep + c0 * (1.0 - keep)
            z_prev_top = z_prev_top * mask_roll.view(BA, 1, 1, 1).to(device=device, dtype=dtype)
            if actions is None:
                a_prev = torch.zeros(BA, self.action_dim, device=device, dtype=dtype)
            else:
                a_prev = actions[:, t_idx_safe - 1].reshape(BA, -1).contiguous()
                a_prev = a_prev * mask_roll.unsqueeze(-1)
            h_tok = self.rnn.out_norm(h)
            B_, _, D_ = h_tok.shape
            h_t = h_tok.transpose(1, 2).reshape(B_, D_, self.top_H, self.top_W ).contiguous()


            kl_ba, _ = compute_slot_kl_conditional_frozen(
                snapshot=self.vdvae.top_prior_snapshot,
                frozen_gate=self.vdvae.top_prior_gate,
                h_t=h_t,
                slot_mu=top_slot_mean[:, t_idx_safe].reshape(BA, S, D).contiguous().detach(),
                slot_logsigma=top_slot_logsigma[:, t_idx_safe].reshape(BA, S, D).contiguous().detach()

            )

            wk = torch.as_tensor(w_decay ** (k - 1), device=device, dtype=dtype)
            numer = numer + wk * (kl_ba * mask_roll).sum()
            denom = denom + wk * mask_roll.sum()

            # 4) imagined top sample
            slot_block = self.vdvae.decoder.dec_blocks[0].top_slot_posterior


            slots, _, _ = sample_slot_vectors_conditional_frozen(
                snapshot=self.vdvae.top_prior_snapshot,
                frozen_gate=self.vdvae.top_prior_gate,
                h_t=h_t,
                num_slots=slot_block.num_slots,
                assignment_temperature=rollout_temperature,
                slot_temperature=rollout_temperature,
            )

            _, _, z_top_state, _, _, _ = slot_block.slot_to_top_prior(slots, h_t, temperature=rollout_temperature)
            # 5) latent 64 update, no decoder
            if actions is None:
                a_cur = torch.zeros(BA, self.action_dim, device=device, dtype=dtype)
            else:
                a_cur = actions[:, t_idx_safe].reshape(BA, -1).contiguous()
                can_step_fwd = (
                    (t_idx_safe < (T - 1))[None, :]
                    .expand(B, A)
                    .reshape(BA)
                    .to(dtype)
                )
                a_cur = a_cur * (mask_roll * can_step_fwd).unsqueeze(-1) #TODO: ???Is this correct?

            _, (h, c) = self.rnn(
                z_top_state,
                a_cur,
                state=(h, c),
                mask_t=None,
                extra_maps=None,
            )
            z_prev_top = z_top_state

        if denom <= 0:
            return torch.zeros((), device=device, dtype=dtype)
        return numer / denom


    @staticmethod
    def orthogonality_loss(slots: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
        """
        slots: [B,S,D] or [B,T,S,D]
        valid_mask: optional [B,T], True for valid frames
        """

        B, T, S, D = slots.shape

        z = F.normalize(slots, p=2, dim=-1, eps=eps)

        gram = z @ z.transpose(-1, -2)  # [B,T,S,S]

        eye = torch.eye(S, device=slots.device, dtype=slots.dtype).view(1, 1, S, S)
        off_diag = gram * (1.0 - eye)

        loss_bt = off_diag.pow(2).sum(dim=(-1, -2)) / (S * (S - 1) + eps)  # [B,T]

        if valid_mask is not None:
            valid = valid_mask.to(device=slots.device, dtype=slots.dtype)
            return (loss_bt * valid).sum() / valid.sum().clamp_min(1.0)

        return loss_bt.mean()
        
    def compute_total_loss(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        beta: float = 1.0,
        lambda_recon: float = 1.0,
        store_reconstruction_samples: bool = False,
        collect_top_buffer: bool = False,
        seq_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.forward_sequence(
            observations,
            actions,
            dones,
            store_reconstruction_samples=store_reconstruction_samples,
            collect_top_buffer=collect_top_buffer,
            seq_ids=seq_ids,
        )
        device = observations.device
        z0 = torch.zeros((), device=device)

        if isinstance(outputs.get("reconstructions", None), list):
            outputs["reconstructions"] = torch.stack(outputs["reconstructions"], dim=1)
        if isinstance(outputs.get("reconstruction_samples", None), list):
            outputs["reconstruction_samples"] = torch.stack(outputs["reconstruction_samples"], dim=1)

        def _mean_seq(key: str):
            xs = outputs.get(key, [])
            if xs is None or len(xs) == 0:
                return z0
            if torch.is_tensor(xs):
                return xs.mean()
            return torch.stack(xs).mean()

        recon_loss = _mean_seq("reconstruction_losses")
        kl_z = _mean_seq("kl_latents")

        slots = outputs["top_slot_mu"]  # [B, T, S, D]
        B_s, T_s, S, D = slots.shape

        alive = None
        if dones is not None:
            dones_for_slots = dones.to(device=slots.device)

            if dones_for_slots.dim() == 1:
                dones_for_slots = dones_for_slots[:, None]

            dones_for_slots = dones_for_slots[:, :T_s]

            alive, _ = self._lengths_from_dones(
                dones_for_slots,
                T_s,
                assume_padded_after_done=True,
            )  # [B,T]

        slot_orth_loss = self.orthogonality_loss(slots, valid_mask=alive)
        slot_contrast_loss = z0

        if self.lambda_slot_contrast > 0.0 and T_s >= 2:
            prev_slots = slots[:, :-1]  # [B, T-1, S, D]
            next_slots = slots[:, 1:]   # [B, T-1, S, D]
            # [B, T-1, 2, S, D]
            two_frame_clips = torch.stack([prev_slots, next_slots], dim=2)

            if alive is not None:
                valid_transition_mask = alive[:, :-1] & alive[:, 1:]  # [B,T-1]
                valid_two_frame_clips = two_frame_clips[valid_transition_mask]
                # [N_valid, 2, S, D]
            else:
                valid_two_frame_clips = two_frame_clips.reshape(
                    B_s * (T_s - 1),
                    2,
                    S,
                    D,
                )

            if valid_two_frame_clips.shape[0] > 0:
                slot_contrast_loss = self.slot_contrast_loss_fn(
                    valid_two_frame_clips,
                    None,
                )
        mask_entropy = z0
        mask_balance = z0

        decoder_masks = outputs["decoder_masks"]

        if (torch.is_tensor(decoder_masks)
            and (self.lambda_mask_entropy > 0.0 or self.lambda_mask_balance > 0.0)
        ):
            # decoder_masks: [B, T, S, 1, H, W]
            if decoder_masks.dim() != 6:
                raise ValueError(f"Expected decoder_masks [B,T,S,1,H,W], got {tuple(decoder_masks.shape)}")

            Bm, Tm, Sm, _, Hm, Wm = decoder_masks.shape

            # [B,T,S,1,H,W] -> [B*T,S,H*W]
            p = decoder_masks.reshape(Bm * Tm, Sm, Hm * Wm)

            # Optional: remove padded/done frames
            if alive is not None:
                valid = alive.reshape(Bm * Tm)
                p = p[valid]

            if p.numel() > 0:
                eps = 1e-8

                # Ensure p(slot | pixel) sums to 1 over slots
                p = p.clamp_min(eps)
                p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)

                # Low entropy => each pixel chooses one/few slots
                mask_entropy = -(p * p.log()).sum(dim=1).mean()

                # Weak global slot usage balance => avoid dead slots
                usage = p.mean(dim=(0, 2))  # [S]
                expected = torch.full_like(usage, 1.0 / float(Sm))
                mask_balance = ((usage - expected) ** 2).mean()
        total_vae_loss = (
            lambda_recon * recon_loss
            + beta * kl_z
            + self.lambda_slot_contrast * slot_contrast_loss
            + self.lambda_slot_orth * slot_orth_loss
            + self.lambda_mask_entropy * mask_entropy
            + self.lambda_mask_balance * mask_balance
        )
        if torch.is_tensor(outputs.get("decoder_masks", None)):
           outputs["decoder_masks"] = outputs["decoder_masks"].detach()
        overshoot_kl = z0
        if getattr(self, "lambda_overshoot", 0.0) > 0.0:
            T = outputs["top_slot_mu"].shape[1]
            K_effect = min(int(self.overshoot_K), max(0, T - 1))
            if K_effect > 0:
                overshoot_kl = self.compute_multistep_kl_overshoot(
                    top_slot_mean=outputs["top_slot_mu"],
                    top_slot_logsigma=outputs["top_slot_logsigma"],
                    actions=actions,
                    dones=dones,
                    K=K_effect,
                    w_decay=self.overshoot_w_decay,
                    core_state_maps=(
                        outputs["core_h_maps"],
                        outputs["core_c_maps"],
                    ),
                    overshoot_anchor_state={
                        "z_prev_top": outputs["overshoot_z_prev_top"],
                    },
                    rollout_temperature=self.rollout_top_temperature,
                )

        total_vae_loss = total_vae_loss + self.lambda_overshoot * overshoot_kl

        vae_losses = {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "overshoot_kl": overshoot_kl,
            "total_vae_loss": total_vae_loss,
            "slot_contrast_loss": slot_contrast_loss,
            "slot_orth_loss": slot_orth_loss,
            "mask_entropy": mask_entropy,
            "mask_balance": mask_balance,
        }
        return vae_losses, outputs

    def compute_gradient_penalty_patch(self, D2d, real_x, fake_x, device, mask_flat=None):
        N = real_x.size(0)
        alpha = torch.FloatTensor(np.random.random((N, 1, 1, 1))).to(device)
        x_hat = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
        d_hat = D2d(x_hat).mean(dim=(1, 2, 3))
        grads = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].view(N, -1)

        gp_per = (grads.norm(2, dim=1) - 1.0).pow(2)
        if mask_flat is None:
            return gp_per.mean()
        return (gp_per * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

    def _make_temporal_mask(self, B: int, T: int, device, sequence_lengths):
        if sequence_lengths is None:
            return None
        t = torch.arange(T, device=device)[None, :]           # [1,T]
        return (t < sequence_lengths[:, None])                # [B,T] bool

    def _masked_mean(self, x, mask):
        # x: [B,T] or [N]
        if mask is None:
            return x.mean()
        mask_f = mask.float()
        return (x * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    def discriminator_step(
        self,
        real_images: torch.Tensor, #[B, T, C, H, W]
        fake_images: torch.Tensor, #[B, T, C, H, W]
        latents: torch.Tensor,  #[B, T, Cz, Ht, Wt]
        sequence_lengths: Optional[torch.Tensor] = None,
        WGAN_GP_Coeff: float = 2.0,
        lambda_consistency: float = 0.4,
        lambda_temporal_lecam: float = 0.2,
        lambda_patch_lecam: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for both discriminators
        """

        B, T, C, H, W = real_images.shape
        temporal_mask = self._make_temporal_mask(B, T, real_images.device, sequence_lengths )         # [B,T] bool
        # For per-frame PatchGAN losses
        mask_flat = None
        if temporal_mask is not None:
            mask_flat = temporal_mask.reshape(B * T).float()  # [B*T]

        disc_losses: Dict[str, torch.Tensor] = {}

        # Temporal Image Discriminator
        real_img_outputs = self.image_discriminator(real_images, z=latents.detach(), mask=temporal_mask)
        fake_img_outputs = self.image_discriminator(fake_images.clamp(-1.0, 1.0).detach(), z=latents.detach(), mask=temporal_mask)

        #Hinge Discriminator Loss
        temporal_disc_loss = (F.relu(1.0 - real_img_outputs['final_score']) + F.relu(1.0 + fake_img_outputs['final_score'])).mean()

        # Temporal consistency losses
        img_consistency_loss = torch.zeros((), device=self.device)
        fake_per_t = fake_img_outputs["per_timestep_score"].squeeze(-1)   # [B,Tdisc]
        if fake_per_t.size(1) > 1:
            pair_mask = None
            if temporal_mask is not None:
                tmask = temporal_mask[:, :fake_per_t.size(1)]
                pair_mask = (tmask[:, 1:] & tmask[:, :-1]).float()
            diffs = (fake_per_t[:, 1:] - fake_per_t[:, :-1]).abs()
            img_consistency_loss = self._masked_mean(diffs, pair_mask)

        real_frames = real_images.reshape(B * T, C, H, W).contiguous()
        fake_frames = fake_images.detach().reshape(B * T, C, H, W).contiguous()

        real_logits = self.patch_discriminator(real_frames)                 # [B*T,1,h,w]
        fake_logits = self.patch_discriminator(fake_frames)                 # [B*T,1,h,w]
        real_frame_score = real_logits.mean(dim=(1,2,3))  # [B*T]
        fake_frame_score = fake_logits.mean(dim=(1,2,3))  # [B*T]

        patch_disc_loss = self._masked_mean(fake_frame_score, mask_flat) - self._masked_mean(real_frame_score, mask_flat)

        patch_gp = self.compute_gradient_penalty_patch(
            D2d=self.patch_discriminator,
            real_x=real_frames,
            fake_x=fake_frames,
            device= real_images.device,
            mask_flat=mask_flat,
        )

        # --- LeCam inputs ---
        real_temporal_for_lc = real_img_outputs["final_score"].reshape(-1)   # one score per sequence
        fake_temporal_for_lc = fake_img_outputs["final_score"].reshape(-1)

        if mask_flat is not None:
            valid = mask_flat > 0.5
            real_patch_for_lc = real_frame_score[valid]
            fake_patch_for_lc = fake_frame_score[valid]
        else:
            real_patch_for_lc = real_frame_score
            fake_patch_for_lc = fake_frame_score

        # current batch means for anchor updates
        temporal_real_mean = real_temporal_for_lc.mean()
        temporal_fake_mean = fake_temporal_for_lc.mean()
        patch_real_mean = real_patch_for_lc.mean()
        patch_fake_mean = fake_patch_for_lc.mean()

        # don't apply LeCam until anchors are initialized
        if bool(self.lecam_initialized):
            temporal_lecam_loss = self.compute_lecam_loss(
                logits_real=real_temporal_for_lc,
                logits_fake=fake_temporal_for_lc,
                ema_logits_real=self.lecam_temporal_real_ema.to(
                    device=real_temporal_for_lc.device, dtype=real_temporal_for_lc.dtype
                ),
                ema_logits_fake=self.lecam_temporal_fake_ema.to(
                    device=fake_temporal_for_lc.device, dtype=fake_temporal_for_lc.dtype
                ),
            )

            patch_lecam_loss = self.compute_lecam_loss(
                logits_real=real_patch_for_lc,
                logits_fake=fake_patch_for_lc,
                ema_logits_real=self.lecam_patch_real_ema.to(
                    device=real_patch_for_lc.device, dtype=real_patch_for_lc.dtype
                ),
                ema_logits_fake=self.lecam_patch_fake_ema.to(
                    device=fake_patch_for_lc.device, dtype=fake_patch_for_lc.dtype
                ),
            )
        else:
            temporal_lecam_loss = real_temporal_for_lc.new_zeros(())
            patch_lecam_loss = real_frame_score.new_zeros(())

        img_disc_loss = temporal_disc_loss + lambda_consistency * img_consistency_loss + patch_disc_loss + WGAN_GP_Coeff * patch_gp + lambda_temporal_lecam * temporal_lecam_loss + lambda_patch_lecam * patch_lecam_loss

        if getattr(self, "img_disc_optimizer", None) is not None:
            # Update image discriminator
            self.img_disc_optimizer.zero_grad()
            img_disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.image_discriminator.parameters()) + list(self.patch_discriminator.parameters()), self._grad_clip, error_if_nonfinite=True)
            self.img_disc_optimizer.step()
            self._update_lecam_ema(
                temporal_real_mean=temporal_real_mean,
                temporal_fake_mean=temporal_fake_mean,
                patch_real_mean=patch_real_mean,
                patch_fake_mean=patch_fake_mean,
            )

        disc_losses.update({
            'img_disc_loss': img_disc_loss.detach(),
            # Temporal discriminator metrics
            'temporal_disc_loss': temporal_disc_loss.detach(),
            'temporal_disc_real': real_img_outputs["final_score"].mean().detach(),
            'temporal_disc_fake': fake_img_outputs["final_score"].mean().detach(),
            'temporal_consistency_loss': img_consistency_loss.detach(),  # Renamed
            # PatchGAN metrics
            'patch_disc_loss': patch_disc_loss.detach(),  # New key
            'patch_gp': patch_gp.detach(),
            'patch_disc_real': real_frame_score.mean().detach(),
            'patch_disc_fake': fake_frame_score.mean().detach(),
            'temporal_lecam_loss': temporal_lecam_loss.detach(),
            'patch_lecam_loss': patch_lecam_loss.detach(),
        })
        return  disc_losses


    def compute_feature_matching_loss(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
        temporal_mask: torch.Tensor | None = None,
    ):
        if temporal_mask is None:
            return F.l1_loss(fake_features, real_features.detach())

        # fake_features, real_features: [B, T, C, H, W]
        diff = torch.abs(fake_features - real_features.detach())
        m = temporal_mask.to(dtype=diff.dtype).view(diff.shape[0], diff.shape[1], 1, 1, 1)

        denom = (m.sum() * diff.shape[2] * diff.shape[3] * diff.shape[4]).clamp(min=1.0)
        return (diff * m).sum() / denom

    def compute_adversarial_losses(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        z_seq: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        lambda_final: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = self.image_discriminator
        Dpatch = self.patch_discriminator
        B, T, C, H, W = reconstruction.shape

        temporal_mask = self._make_temporal_mask(B, T, reconstruction.device, sequence_lengths)
        mask_flat = None
        if temporal_mask is not None:
            mask_flat = temporal_mask.reshape(B * T).float()

        flags = [p.requires_grad for p in D.parameters()]
        for p in D.parameters():
            p.requires_grad_(False)

        fake_img_outputs = D(
            reconstruction.clamp(-1.0, 1.0),
            z=z_seq,
            mask=temporal_mask,
            return_features=True
        )
        real_img_outputs = D(
            x,
            z=z_seq.detach(),
            mask=temporal_mask,
            return_features=True
        )

        final_scores = fake_img_outputs["final_score"].squeeze(-1).squeeze(-1)
        final_adv_loss = -final_scores.mean()
        temporal_adv_loss = lambda_final * final_adv_loss

        mask_tdisc = None
        if temporal_mask is not None:
            mask_tdisc = temporal_mask[:, :fake_img_outputs["frame_features"].shape[1]]

        feature_match_loss = self.compute_feature_matching_loss(
            real_features=real_img_outputs["frame_features"],
            fake_features=fake_img_outputs["frame_features"],
            temporal_mask=mask_tdisc,
        )

        for p, f in zip(D.parameters(), flags):
            p.requires_grad_(f)

        patch_flags = [p.requires_grad for p in Dpatch.parameters()]
        for p in Dpatch.parameters():
            p.requires_grad_(False)

        fake_frames = reconstruction.clamp(-1.0, 1.0).reshape(B * T, C, H, W).contiguous()
        with torch.autocast(device_type="cuda", enabled=False):
            patch_logits = Dpatch(fake_frames.float())
        patch_scores = patch_logits.mean(dim=(1, 2, 3))
        img_adv_loss = -self._masked_mean(patch_scores, mask_flat)

        for p, f in zip(Dpatch.parameters(), patch_flags):
            p.requires_grad_(f)

        return img_adv_loss, temporal_adv_loss, feature_match_loss

    def denormalize_generated_images(self, images):
        """
        Convert generated images from [-1, 1] back to [0, 1] for visualization
        """
        return (images + 1) / 2


    def training_step_sequence(self,
                            observations: torch.Tensor,
                            actions: torch.Tensor = None,
                            dones: torch.Tensor = None,
                            beta: float = 1.0,
                            n_critic: int = 3,
                            lambda_img: float = 0.5,
                            lambda_recon: float = 1.0,
                            batch_idx: Optional[int] = None,
                            collect_top_buffer: bool = False,
                            seq_ids: Optional[torch.Tensor] = None,
                            ) -> Dict[str, torch.Tensor]:
        """
        Dynamic Weight Averaging (DWA) or fix loss coefficients

        """
        self.train()
        if getattr(self, "image_discriminator", None) is not None:
            self.image_discriminator.train()
        # 1) Prepare data and compute VAE loss
        warmup_factor = self.get_warmup_factor()
        vae_losses, outputs = self.compute_total_loss(
            observations,
            actions,
            dones,
            beta,
            lambda_recon,
            collect_top_buffer=collect_top_buffer,
            seq_ids=seq_ids,
        )
        # z_seq for disc conditioning (teacher-forced)
        z_seq_tf = outputs["z_seq_maps"]  # [B, T, z+h, Ht, Wt]
        B, seq_len = observations.shape[:2]
        if dones is not None:
            _, lengths_full = self._lengths_from_dones(dones, T=seq_len, assume_padded_after_done=True)
        else:
            lengths_full = torch.full((B,), seq_len, device=observations.device, dtype=torch.long)

        # 2) Decide whether to do rollout GAN this step
        T_total = seq_len
        do_rollout = (
            self.lambda_rollout_adv > 0.0
            and self.rollout_adv_every > 0
            and (self.global_step % self.rollout_adv_every == 0)
            and (T_total >= 2)
        )

        rollout_horizon = 0
        T_ctx = 0
        real_future = None
        fake_future_D = None
        z_seq_roll_D = None
        seq_len_future = None
        keep = None
        actions_slice = None
        dones_slice = None

        if do_rollout:
            T_ctx = min(self.rollout_context_frames, T_total - 1)
            rollout_horizon = min(self.rollout_horizon, T_total - T_ctx)

            if rollout_horizon <= 0:
                do_rollout = False
            else:
                real_future = observations[:, T_ctx:T_ctx + rollout_horizon]
                actions_slice = actions[:, :T_ctx + rollout_horizon] if actions is not None else None
                dones_slice = dones[:, :T_ctx + rollout_horizon] if dones is not None else None

                if dones_slice is not None:
                    alive_slice, _ = self._lengths_from_dones(
                        dones_slice, T_ctx + rollout_horizon, assume_padded_after_done=True
                    )
                    future_alive = alive_slice[:, T_ctx:T_ctx + rollout_horizon]   # [B, H]
                    future_len = future_alive.long().sum(dim=1)                    # [B]
                    keep = (future_len > 0)
                else:
                    keep = torch.ones(B, device=observations.device, dtype=torch.bool)
                    future_len = torch.full((B,), rollout_horizon, device=observations.device, dtype=torch.long)

                if keep.sum().item() == 0:
                    do_rollout = False
                    real_future = None
                else:
                    # Apply keep consistently
                    real_future = real_future[keep]
                    seq_len_future = future_len[keep]

                    # Rollout once WITHOUT grad for discriminator updates
                    
                    dbgD = self.generate_future_sequence(
                        initial_obs=observations[keep, :T_ctx],
                        actions=(actions_slice[keep] if actions_slice is not None else None),
                        horizon=rollout_horizon,
                        top_temperature=self.rollout_top_temperature,
                        dones=(dones_slice[keep] if dones_slice is not None else None),
                        grad=False,
                    )
                    fake_future_D = dbgD["vae_future"]   # [B_keep, H, C, H, W]
                    z_seq_roll_D = dbgD["z_seq"]         # [B_keep, H, Z] (should already be detached inside)

        # 3) Discriminator updates (n_critic)
        disc_losses_list: List[Dict[str, torch.Tensor]] = []
        for _ in range(n_critic):
            # teacher-forced recon fakes
            disc_loss = self.discriminator_step(
                real_images=observations.float(),
                fake_images=outputs["reconstructions"].detach().float(),
                latents=z_seq_tf.detach().float(),
                sequence_lengths=lengths_full,
            )

            # rollout fakes (optional)
            if do_rollout and (fake_future_D is not None) and (z_seq_roll_D is not None):
                disc_loss_roll = self.discriminator_step(
                    real_images=real_future.float(),
                    fake_images=fake_future_D.detach().float(),
                    latents=z_seq_roll_D.detach().float(),
                    sequence_lengths=seq_len_future,
                )
                for k, v in disc_loss_roll.items():
                    disc_loss[f"rollout_{k}"] = v

            disc_losses_list.append(disc_loss)

        avg_disc_losses: Dict[str, torch.Tensor] = {}
        if disc_losses_list:
            avg_disc_losses = {
                k: sum(d[k] for d in disc_losses_list) / len(disc_losses_list)
                for k in disc_losses_list[0].keys()
            }

        # 4) Generator adversarial losses (teacher-forced)
        lambda_img_eff = (lambda_img * warmup_factor) if warmup_factor > 0.0 else 0.0

        img_adv_loss, temporal_adv_loss, feat_match_loss = self.compute_adversarial_losses(
            x=observations,
            reconstruction=outputs["reconstructions"],
            z_seq=z_seq_tf,
            sequence_lengths=lengths_full
        )

        # 5) Generator adversarial losses (rollout) — SAVP-style prior realism
        rollout_img_adv_loss = torch.zeros((), device=observations.device)
        rollout_temporal_adv_loss = torch.zeros((), device=observations.device)
        rollout_feat_match_loss = torch.zeros((), device=observations.device)

        if do_rollout and rollout_horizon > 0:
            # Rollout again WITH grad for generator update
            dbgG = self.generate_future_sequence(
                initial_obs=observations[keep, :T_ctx],
                actions=(actions_slice[keep] if actions_slice is not None else None),
                horizon=rollout_horizon,
                top_temperature=self.rollout_top_temperature,
                dones=(dones_slice[keep] if dones_slice is not None else None),
                grad=True,
            )
            fake_future_G = dbgG["vae_future"]      # [B_keep, H, C, H, W] (requires grad)
            z_seq_roll_G = dbgG["z_seq"].detach()   # keep conditioning stable + save memory

            rollout_img_adv_loss, rollout_temporal_adv_loss, rollout_feat_match_loss = self.compute_adversarial_losses(
                x=real_future,
                reconstruction=fake_future_G,
                z_seq=z_seq_roll_G,
                sequence_lengths=seq_len_future,
            )
            # Flatten time for LPIPS
            fake_bt = rearrange(fake_future_G, "b t c h w -> (b t) c h w")
            real_bt = rearrange(real_future,   "b t c h w -> (b t) c h w")

            # --- Edge consistency loss (rollout) ---
            # Optional: work in [0,1] for more stable edge magnitudes
            fake01 = self.denormalize_generated_images(fake_future_G)  # [-1,1] -> [0,1]
            real01 = self.denormalize_generated_images(real_future)

            Bf, Tf, C, H, W = fake01.shape
            fake_flat = fake01.reshape(Bf * Tf, C, H, W)
            real_flat = real01.reshape(Bf * Tf, C, H, W)

        # 6) Combine losses with DWA or fixed weights
        if self.use_dwa:
            total_components = {
                "recon_loss": vae_losses["recon_loss"].reshape([]),
                "kl_z": vae_losses["kl_z"].reshape([]),
                "img_adv_loss": (warmup_factor * img_adv_loss).reshape([]),
                "temporal_adv_loss": (warmup_factor * temporal_adv_loss).reshape([]),
                "feat_match_loss": (warmup_factor * feat_match_loss).reshape([]),
                "slot_contrast_loss": vae_losses["slot_contrast_loss"].reshape([]),
                "slot_orth_loss": vae_losses["slot_orth_loss"].reshape([]),
                "mask_entropy": vae_losses["mask_entropy"].reshape([]),
                "mask_balance": vae_losses["mask_balance"].reshape([]),
                "rollout_img_adv_loss": (warmup_factor * rollout_img_adv_loss).reshape([]),
                "rollout_temporal_adv_loss": ( warmup_factor * rollout_temporal_adv_loss).reshape([]),
                "rollout_feat_match_loss": ( warmup_factor * rollout_feat_match_loss).reshape([]),


                "overshoot_kl": vae_losses["overshoot_kl"].reshape([]),
            }

            total_gen_loss = self.total_weighter.reduce_losses(total_components, batch_idx)
        else:
            adv_base = (
                lambda_img_eff * img_adv_loss
                + warmup_factor * temporal_adv_loss
                + warmup_factor * lambda_img * feat_match_loss
            )
            adv_roll = self.lambda_rollout_adv * (
                lambda_img_eff * rollout_img_adv_loss
                + warmup_factor * rollout_temporal_adv_loss
                + warmup_factor * lambda_img * rollout_feat_match_loss
            )
            total_gen_loss = (
                vae_losses["total_vae_loss"]
                + adv_base
                + adv_roll
            )

        # 7) Backprop generator
        self.gen_optimizer.zero_grad(set_to_none=True)
        total_gen_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for g in self.gen_optimizer.param_groups for p in g["params"]],
            self._grad_clip,
        )
        self.gen_optimizer.step()
        self.global_step += 1
        grad_norm_sq = 0.0
        for group in self.gen_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = p.grad.data
                    grad_norm_sq = grad_norm_sq + float((g.norm(2)).item() ** 2)
        grad_norm = (grad_norm_sq ** 0.5) if grad_norm_sq > 0 else 0.0

        # 8) EMA updates for encoder/decoder
        with torch.no_grad():
            self.ema_vdvae.update()

        pi_seq = torch.stack(outputs["prior_pi"], dim=1)

        eff_comp = pi_seq.max(dim=-1).values.mean(dim=0).mean().item()                 # [T]
        top6_cov = pi_seq.topk(min(6, pi_seq.size(-1)), dim=-1).values.sum(dim=-1).mean(dim=0).mean().item()  # [T]

        return {
            **vae_losses,
            **avg_disc_losses,

            "img_adv_loss": float(img_adv_loss.item()),
            "temporal_adv_loss": float(temporal_adv_loss.item()),
            "feat_match_loss": float(feat_match_loss.item()),
            "rollout_img_adv_loss": float(rollout_img_adv_loss.item()),
            "rollout_temporal_adv_loss": float(rollout_temporal_adv_loss.item()),
            "rollout_feat_match_loss": float(rollout_feat_match_loss.item()),
            "did_rollout_adv": float(1.0 if do_rollout else 0.0),
            "total_gen_loss": float(total_gen_loss.item()),
            "grad_norm": float(grad_norm),
            "effective_components": eff_comp,
            "Top 6 coverage": top6_cov,
        }


    def get_warmup_factor(self) -> float:
        """Calculate warmup factor for adversarial and flow-alignment losses."""
        if self.current_epoch < self.warmup_epochs:
            return self.current_epoch / self.warmup_epochs
        return 1.0


    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        device = self.device
        dtype = next(self.parameters()).dtype

        core_state = self.rnn.init_state(num_samples, device=device, dtype=dtype)
        h_tok = self.rnn.out_norm(core_state[0])
        B_, _, D_ = h_tok.shape
        h_t = h_tok.transpose(1, 2).reshape(B_, D_, self.top_H, self.top_W ).contiguous()
        z_prev_top = torch.zeros(num_samples, self.zdim, self.top_H, self.top_W, device=device, dtype=dtype)



        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.apply_shadow()
        prev_latents = self._init_decoder_prev_latents(num_samples, device, dtype)

        x_np, current_latents = self.vdvae.sample(
            num_samples,
            h_prior_top=h_t,
            prev_latents=prev_latents,
        )

        if hasattr(self, "ema_vdvae"):
            self.ema_vdvae.restore()

        x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous().float() / 127.5 - 1.0
        return x

    def generate_future_sequence(
        self,
        initial_obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        horizon: int = 15,
        top_temperature: float = 1.0,
        dones: Optional[torch.Tensor] = None,
        grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Warm up on context frames with teacher forcing, then roll forward by sampling
        one full top latent map per image from the image-level DP-GMM prior.

        Returns:
            vae_future: [B, H_roll, C, H, W]
            z_seq:      [B, H_roll, zdim + self.latent_dim + self.action_dim, top_H, top_W]
            pi_seq:     [B, H_roll, K]
        """
        B, T_ctx, C, H, W = initial_obs.shape
        device = initial_obs.device
        dtype = initial_obs.dtype

        if actions is None:
            actions = torch.zeros(
                B, T_ctx + horizon, self.action_dim, device=device, dtype=dtype
            )
        elif actions.shape[1] < (T_ctx + horizon):
            pad = (T_ctx + horizon) - actions.shape[1]
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(B, pad, actions.shape[2], device=device, dtype=actions.dtype),
                ],
                dim=1,
            )

        if dones is not None and dones.shape[1] < (T_ctx + horizon):
            pad = (T_ctx + horizon) - dones.shape[1]
            dones = torch.cat(
                [dones, torch.zeros(B, pad, device=device, dtype=dones.dtype)],
                dim=1,
            )

        def mask_at(t_abs: int) -> torch.Tensor:
            if dones is None or t_abs == 0:
                return torch.ones(B, device=device, dtype=torch.float32)
            return (1.0 - dones[:, t_abs - 1].float()).to(torch.float32)

        core_state = self.rnn.init_state(B, device=device, dtype=dtype)
        z_prev_top = torch.zeros(B, self.zdim, self.top_H, self.top_W, device=device, dtype=dtype)
        prev_latents = self._init_decoder_prev_latents(B, device, dtype)

        pred_imgs: List[torch.Tensor] = []
        z_seq: List[torch.Tensor] = []
        pi_seq: List[torch.Tensor] = []

        with torch.set_grad_enabled(grad):
            # ---------------------------------
            # teacher-forced warmup on context
            # ---------------------------------
            for t in range(T_ctx):
                x_t = initial_obs[:, t]
                mask_t = mask_at(t)

                keep = mask_t.view(B, 1, 1).to(device=device, dtype=dtype)
                h, c = core_state
                h0, c0= self.rnn.init_state(B, device=device, dtype=dtype)
                core_state = (
                    h * keep + h0 * (1.0 - keep),
                    c * keep + c0 * (1.0 - keep),
                )

                z_prev_top_masked = z_prev_top * mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype)
                assert all(pl is None or pl.shape[0] == B or pl.shape[0] % B == 0 for pl in prev_latents)
                prev_latents = [None if pl is None else pl * (mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype) if pl.shape[0] == B else mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype).repeat_interleave(pl.shape[0] // B, dim=0)) for pl in prev_latents]

                h_tok = self.rnn.out_norm(core_state[0])
                B_, _, D_ = h_tok.shape
                h_t = h_tok.transpose(1, 2).reshape(B_, D_, self.top_H, self.top_W ).contiguous()


                x_t_nhwc = x_t.permute(0, 2, 3, 1).contiguous()
                vdvae_out = self.vdvae.forward(
                    x_t_nhwc,
                    x_t_nhwc,
                    h_prior_top=h_t,
                    mask_t=mask_t,
                    prev_latents=prev_latents,
                    get_latents=True,
                )
                z_top_state = vdvae_out.get("z_top_state", vdvae_out["current_latents"][0])

                a_cur = actions[:, t]
                _, core_state = self.rnn(
                    z_top_state,
                    a_cur,
                    state=core_state,
                    mask_t=None,
                    extra_maps=None,
                )
                core_state = tuple(s.detach() for s in core_state)
                z_prev_top = z_top_state.detach()
                prev_latents = [None if idx == 0 else pl.detach() for idx, pl in enumerate(vdvae_out["current_latents"])]

            # future rollout from prior
            for k in range(horizon):
                t_abs = T_ctx + k
                mask_t = mask_at(t_abs)

                keep = mask_t.view(B, 1, 1).to(device=device, dtype=dtype)
                h, c= core_state
                h0, c0 = self.rnn.init_state(B, device=device, dtype=dtype)
                core_state = (
                    h * keep + h0 * (1.0 - keep),
                    c * keep + c0 * (1.0 - keep),
                )

                z_prev_top_masked = z_prev_top * mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype)
                assert all(pl is None or pl.shape[0] == B or pl.shape[0] % B == 0 for pl in prev_latents)
                prev_latents = [None if pl is None else pl * (mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype) if pl.shape[0] == B else mask_t.view(B, 1, 1, 1).to(device=device, dtype=dtype).repeat_interleave(pl.shape[0] // B, dim=0)) for pl in prev_latents]

                h_tok = self.rnn.out_norm(core_state[0])
                B_, _, D_ = h_tok.shape
                h_t = h_tok.transpose(1, 2).reshape(B_, D_, self.top_H, self.top_W ).contiguous()

                a_prev = (
                    actions[:, t_abs - 1] * mask_t.view(B, 1).to(device=device, dtype=dtype)
                    if t_abs > 0
                    else torch.zeros(B, self.action_dim, device=device, dtype=dtype)
                )


                # frozen conditional mixture weights p(z_t^top | h_t)
                slot_block = self.vdvae.decoder.dec_blocks[0].top_slot_posterior

                slots, _, _ = sample_slot_vectors_conditional_frozen(
                    snapshot=self.vdvae.top_prior_snapshot,
                    frozen_gate=self.vdvae.top_prior_gate,
                    h_t=h_t,
                    num_slots=slot_block.num_slots,
                    assignment_temperature=top_temperature,
                    slot_temperature=top_temperature,
                )

                _, _, z_top_state, _, _, z_slot_maps = slot_block.slot_to_top_prior(slots, h_t, temperature=top_temperature)
                elogpi_t = ConditionalTopDPGMM.conditional_expected_log_pi_frozen(
                                self.vdvae.top_prior_gate,
                                h_t,
                            )
                pi_seq.append(torch.softmax(elogpi_t / top_temperature, dim=1).detach())
                px_z, current_latents = self.vdvae.decode_from_top_latent(
                    z_slot_maps,
                    h_decoder_top=h_t,
                    t= top_temperature,
                    prev_latents=prev_latents,
                )

                if grad:
                    dmol_out = self.vdvae.decoder.out_net.forward(px_z)
                    rgb_slots = mean_from_discretized_mix_logistic(
                        dmol_out,
                        self.vdvae.H.num_mixtures,
                    )  # [B*S,H,W,3]

                    N, H_img, W_img, _ = rgb_slots.shape
                    if N % B != 0:
                        raise ValueError(f"Expected rollout px_z batch B*S, got N={N}, B={B}")

                    S = N // B

                    rgb_slots = rgb_slots.view(B, S, H_img, W_img, 3).permute(0, 1, 4, 2, 3)

                    mask_logits = px_z[:, self.vdvae.H.width:self.vdvae.H.width + 1]
                    mask_logits = mask_logits.view(B, S, 1, H_img, W_img)
                    decoder_masks = F.softmax(mask_logits, dim=1)

                    x_hat = torch.sum(decoder_masks * rgb_slots, dim=1)
                    # [B,3,H,W]

                else:
                    x_np = self.vdvae.decoder.out_net.sample(px_z)
                    x_hat = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
                    x_hat = x_hat.permute(0, 3, 1, 2).contiguous() / 127.5 - 1.0

                    x_hat = x_hat.clamp(-1.0, 1.0)
                pred_imgs.append(x_hat)

                a_cur = (
                    actions[:, t_abs]
                    if t_abs < (T_ctx + horizon - 1)
                    else torch.zeros_like(a_prev)
                )

                h_next, core_state = self.rnn(
                    z_top_state,
                    a_cur,
                    state=core_state,
                    mask_t=None,
                    extra_maps=None,
                )

                # preserve spatial structure for discriminator / analysis
                z_seq.append(torch.cat([z_top_state, h_next], dim=1).detach())
                z_prev_top = z_top_state.detach()
                prev_latents = [None if idx == 0 else pl.detach() for idx, pl in enumerate(current_latents)]

        vae_future = (
            torch.stack(pred_imgs, dim=1)
            if pred_imgs
            else initial_obs.new_zeros((B, 0, C, H, W))
        )

        z_seq_out = (
            torch.stack(z_seq, dim=1)
            if z_seq
            else initial_obs.new_zeros(
                (B, 0, self.zdim + self.latent_dim + self.action_dim, self.top_H, self.top_W)
            )
        )

        pi_seq_out = (
            torch.stack(pi_seq, dim=1)
            if pi_seq
            else initial_obs.new_zeros((B, 0, self.max_K))
        )

        return {
            "vae_future": vae_future,
            "z_seq": z_seq_out,
            "pi_seq": pi_seq_out,
        }
