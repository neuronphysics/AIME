import os
import random
from typing import Optional, Dict, List
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.models import inception_v3, Inception_V3_Weights
import cv2
from torchvision import transforms
import wandb
import lpips
from tqdm.auto import tqdm
from einops import rearrange
from VRNN.perceiver.video_prediction_perceiverIO import VQPTTokenizer
from torch.utils.tensorboard import SummaryWriter
from VRNN.dmc_vb_transition_dynamics_trainer import (
    DMCVBDataset,
    safe_collate,
)
from pathlib import Path
import os
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

print("Script dir:", SCRIPT_DIR)
print("Parent dir:", PARENT_DIR)
# SSIM (your implementation) + batch wrapper
def calculate_ssim(img1, img2):
    """Single-image SSIM on HWC uint8 in [0,255]."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim_batch(
    real: torch.Tensor,
    pred: torch.Tensor,
    max_workers: Optional[int] = None,
) -> float:
    """
    real, pred: [N, C, H, W] in [0,1].
    Returns mean SSIM over N images.
    """
    from concurrent.futures import ThreadPoolExecutor

    assert real.shape == pred.shape, "SSIM: real and pred must have same shape"
    assert real.dim() == 4, "SSIM expects tensors of shape [N, C, H, W]"

    real_np = real.detach().clamp(0, 1).cpu().numpy()
    pred_np = pred.detach().clamp(0, 1).cpu().numpy()

    real_np = (real_np * 255.0).astype(np.uint8)
    pred_np = (pred_np * 255.0).astype(np.uint8)

    real_np = np.transpose(real_np, (0, 2, 3, 1))
    pred_np = np.transpose(pred_np, (0, 2, 3, 1))

    N = real_np.shape[0]
    pairs = [(real_np[i], pred_np[i]) for i in range(N)]

    def _worker(pair):
        img1, img2 = pair
        return calculate_ssim(img1, img2)

    if max_workers is None or max_workers <= 1:
        scores = [_worker(p) for p in pairs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            scores = list(ex.map(_worker, pairs))

    return float(np.mean(scores))


# calculate PSNR for evaluation
def calculate_psnr_batch(real: torch.Tensor, pred: torch.Tensor) -> float:
    """
    real, pred: [N, C, H, W] in [0,1].
    Returns mean PSNR over N images.
    """
    eps = 1e-8
    mse = F.mse_loss(pred, real, reduction="none")  # [N,C,H,W]
    mse = mse.view(mse.size(0), -1).mean(dim=1)     # [N]
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))    # [N]
    return float(psnr.mean().item())


# Simple FID implementation using InceptionV3 features
def build_inception(device: torch.device):
    weights = Inception_V3_Weights.IMAGENET1K_V1  # or Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def get_inception_activations(
    imgs: torch.Tensor,  # [N,3,H,W], in [0,1]
    model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    feats = []
    N = imgs.size(0)
    for i in range(0, N, batch_size):
        batch = imgs[i : i + batch_size].to(device)
        # Resize to 299x299
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        f = model(batch)  # [B,2048]
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)  # [N,2048]


def calculate_fid_from_activations(act1: torch.Tensor, act2: torch.Tensor) -> float:
    """
    act1, act2: [N, D] feature tensors (on CPU).
    """
    mu1 = act1.mean(dim=0)
    mu2 = act2.mean(dim=0)

    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    # Covariances
    cov1 = torch.cov(act1.T)
    cov2 = torch.cov(act2.T)

    # sqrtm of cov1 @ cov2 via eigen-decomposition
    cov_prod = cov1 @ cov2
    eigvals, eigvecs = torch.linalg.eigh(cov_prod)
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    sqrt_cov_prod = (eigvecs * eigvals_clamped.sqrt()) @ eigvecs.T

    trace = torch.trace(cov1 + cov2 - 2.0 * sqrt_cov_prod)

    fid = diff_sq + trace
    return float(fid.item())


def compute_fid(
    real_imgs: torch.Tensor,   # [N,3,H,W] in [0,1]
    fake_imgs: torch.Tensor,   # [N,3,H,W] in [0,1]
    device: torch.device,
    inception: Optional[nn.Module] = None,
) -> float:
    if inception is None:
        inception = build_inception(device)

    real_acts = get_inception_activations(real_imgs, inception, device)
    fake_acts = get_inception_activations(fake_imgs, inception, device)

    return calculate_fid_from_activations(real_acts, fake_acts)


# Random block-masking augmentation (for contrastive)
def random_block_mask(
    videos: torch.Tensor,
    mask_prob: float = 0.5,
    min_frac: float = 0.1,
    max_frac: float = 0.4,
) -> torch.Tensor:
    """
    videos: [B, T, C, H, W] in [-1,1]
    Returns a masked copy.
    """
    if mask_prob <= 0.0:
        return videos

    B, T, C, H, W = videos.shape
    videos_masked = videos.clone()

    for b in range(B):
        for t in range(T):
            if random.random() < mask_prob:
                h_frac = random.uniform(min_frac, max_frac)
                w_frac = random.uniform(min_frac, max_frac)
                h_size = max(1, int(H * h_frac))
                w_size = max(1, int(W * w_frac))
                y0 = random.randint(0, H - h_size)
                x0 = random.randint(0, W - w_size)
                # Zero out patch (0 ~ gray in [-1,1] scale)
                videos_masked[b, t, :, y0 : y0 + h_size, x0 : x0 + w_size] = 0.0

    return videos_masked


# Contrastive InfoNCE on latent features
def extract_video_repr(tokenizer: VQPTTokenizer, videos: torch.Tensor) -> torch.Tensor:
    """
    videos: [B,T,C,H,W] in [-1,1].
    Returns [B,D] normalized representation using DVAE encoder logits.
    """
    # DVAE expects [B,C,T,H,W]
    x = videos.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
    logits = tokenizer.dvae._encode_logits(x)  # [B,z_dim,T',H',W']
    # Global average over time & space
    feats = logits.mean(dim=(2, 3, 4))  # [B,z_dim]
    feats = F.normalize(feats, dim=-1)
    return feats

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    z1, z2: [B,D], normalized.
    """
    B, D = z1.shape
    logits = (z1 @ z2.t()) / temperature  # [B,B]
    labels = torch.arange(B, device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


# Save reconstruction grid (original vs recon)
@torch.no_grad()
def save_reconstruction_grid(
    tokenizer: VQPTTokenizer,
    data_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epoch: int,
    num_samples: int = 8,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    writer: Optional[SummaryWriter] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.eval()
    batch = next(iter(data_loader))
    x = batch["observations"].to(device)  # [B,T,C,H,W] in [-1,1]
    x = x[:num_samples]
    token_ids, quantized, _, _ = tokenizer.encode(x)
    recon = tokenizer.decode(quantized)  # [-1,1]

    # one frame per sequence for visualization
    x_vis = x[:, 0]      # [B, C, H, W]
    recon_vis = recon[:, 0]

    x_vis = ((x_vis + 1.0) / 2.0).clamp(0, 1)
    recon_vis = ((recon_vis + 1.0) / 2.0).clamp(0, 1)

    grid = make_grid(
        torch.cat([x_vis, recon_vis], dim=0),
        nrow=num_samples,
        padding=2,
    )
    save_path = os.path.join(out_dir, f"recon_epoch_{epoch:04d}.png")
    save_image(grid, save_path)

    if wandb_run is not None:
        wandb.log({"reconstruction": wandb.Image(save_path)}, step=epoch)

    if writer is not None:
        # grid is [C,H,W], in [0,1]
        writer.add_image("reconstruction", grid, global_step=epoch)

    print(f"[recon] saved grid to {save_path}")


# Evaluation: SSIM, PSNR, FID
@torch.no_grad()
def evaluate_tokenizer(
    tokenizer: VQPTTokenizer,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
    max_fid_images: int = 2048,
    inception: Optional[nn.Module] = None,
) -> Dict[str, float]:
    tokenizer.eval()
    mse_list: List[float] = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []

    real_for_fid: List[torch.Tensor] = []
    recon_for_fid: List[torch.Tensor] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(data_loader):
            if b_idx >= max_batches:
                break

            x = batch["observations"].to(device)  # [B,T,C,H,W] in [-1,1]
            token_ids, quantized, _, _ = tokenizer.encode(x)
            recon = tokenizer.decode(quantized)  # [-1,1]

            # use first frame per sequence for metrics
            x0 = x[:, 0]
            recon0 = recon[:, 0]

            x0_01 = ((x0 + 1.0) / 2.0).clamp(0, 1)
            recon0_01 = ((recon0 + 1.0) / 2.0).clamp(0, 1)

            # MSE
            mse = F.mse_loss(recon0_01, x0_01).item()
            mse_list.append(mse)

            # PSNR
            psnr = calculate_psnr_batch(x0_01, recon0_01)
            psnr_list.append(psnr)

            # SSIM
            ssim = calculate_ssim_batch(x0_01, recon0_01, max_workers=8)
            ssim_list.append(ssim)

            # Collect images for FID
            if len(real_for_fid) * x0_01.size(0) < max_fid_images:
                real_for_fid.append(x0_01.cpu())
                recon_for_fid.append(recon0_01.cpu())

    metrics = {
        "mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "psnr": float(np.mean(psnr_list)) if psnr_list else float("nan"),
        "ssim": float(np.mean(ssim_list)) if ssim_list else float("nan"),
        "fid": float("nan"),
    }

    if real_for_fid:
        real_imgs = torch.cat(real_for_fid, dim=0)  # [N,3,H,W]
        recon_imgs = torch.cat(recon_for_fid, dim=0)
        N = min(real_imgs.size(0), max_fid_images)
        real_imgs = real_imgs[:N]
        recon_imgs = recon_imgs[:N]
        fid = compute_fid(real_imgs, recon_imgs, device=device, inception=inception)
        metrics["fid"] = fid

    return metrics


# Training loop
def train_vqpt_tokenizer_dmc_vb(config: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def center_crop_sample(sample):
        """
        Apply your ADM-style center_crop_arr to each frame in sample['observations'].

        sample['observations']: [T, C, H, W] in [-1, 1]
        After this: still [T, C, H, W] in [-1, 1], but center-cropped/resized
        to (config['img_height'], config['img_width']).
        """
        obs = sample["observations"]  # [T, C, H, W]
        T, C, H, W = obs.shape
        target_size = config["img_height"]  # assume square

        frames = []
        for t in range(T):
            frame = obs[t]  # [C, H, W], in [-1, 1]

            # [-1,1] -> [0,1]
            frame_01 = ((frame + 1.0) / 2.0).clamp(0, 1)

            # [0,1] -> uint8 HWC for PIL
            frame_np = (frame_01.detach().cpu().numpy() * 255.0).astype(np.uint8)
            frame_np = np.transpose(frame_np, (1, 2, 0))  # [H, W, C]

            pil_img = Image.fromarray(frame_np)
            pil_cropped = center_crop_arr(pil_img, target_size)

            # Back to float tensor in [-1,1]
            cropped_np = np.array(pil_cropped).astype(np.float32) / 255.0  # [0,1]
            cropped_np = (cropped_np * 2.0) - 1.0  # [-1,1]
            cropped_np = np.transpose(cropped_np, (2, 0, 1))  # [C, H, W]

            frames.append(torch.from_numpy(cropped_np).to(obs.device))

        sample["observations"] = torch.stack(frames, dim=0)  # [T, C, H, W]
        return sample

    # --- DATASETS ---
    

    train_dataset = DMCVBDataset(
        data_dir=config["data_dir"],
        domain_name=config.get("domain_name", "humanoid"),
        task_name=config.get("task_name", "walk"),
        policy_level="expert",
        split="train",
        sequence_length=config.get("sequence_length", 1),
        frame_stack=config.get("frame_stack", 1),
        img_height=config.get("img_height", 64),
        img_width=config.get("img_width", 64),
        normalize_images=True,
        add_state=False,
        add_rewards=False,
        transform=center_crop_sample,
    )

    val_dataset = DMCVBDataset(
        data_dir=config["data_dir"],
        domain_name=config.get("domain_name", "humanoid"),
        task_name=config.get("task_name", "walk"),
        policy_level="expert",
        split="eval",
        sequence_length=config.get("sequence_length", 1),
        frame_stack=config.get("frame_stack", 1),
        img_height=config["img_height"],
        img_width=config["img_width"],
        normalize_images=True,
        add_state=False,
        add_rewards=False,
        transform=center_crop_sample,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=safe_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=safe_collate,
        drop_last=False,
    )

    # --- MODEL ---
    tokenizer = VQPTTokenizer(
            in_channels=config["in_channels"],
            code_dim=config["code_dim"],
            num_codes=config["num_codes"],
            downsample=config["downsample"],
            base_channels=config["base_channels"],
            commitment_weight=config["commitment_weight"],
            use_cosine_sim=False,
            kmeans_init=config["kmeans_init"],
            dropout=config["dropout"],
            num_quantizers=config["num_quantizers"],
            commitment_use_cross_entropy_loss=config["commitment_use_cross_entropy_loss"],
            orthogonal_reg_weight=config["orthogonal_reg_weight"],
            orthogonal_reg_active_codes_only=config["orthogonal_reg_active_codes_only"],
            orthogonal_reg_max_codes=config["orthogonal_reg_max_codes"],
            codebook_diversity_loss_weight=config["codebook_diversity_loss_weight"],
            codebook_diversity_temperature=config["codebook_diversity_temperature"],
            threshold_ema_dead_code=config["threshold_ema_dead_code"]
    ).to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        tokenizer.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"],
    )

    # --- Logging backends ---
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
    writer: Optional[SummaryWriter] = None

    out_dir = config["out_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Choose logging backend
    if config.get("use_wandb", False) and config.get("wandb_project"):
        # you should run `wandb login` once in your shell
        wandb.login()
        wandb_run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=config["wandb_run_name"],
            config=config,
        )
        wandb.watch(tokenizer, log="all", log_freq=100)
    elif config.get("use_tensorboard", False):
        tb_log_dir = os.path.join(out_dir, "tb_logs")
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[logging] Using TensorBoard at {tb_log_dir}")


    contrastive_weight = config["contrastive_weight"]
    mask_prob = config["mask_prob"]


    inception = build_inception(device)

    global_step = 0
    for epoch in range(1, config.get("epochs", 50) + 1):
        tokenizer.train()
        running_recon = 0.0
        running_vq = 0.0
        running_cl = 0.0
        count = 0
        pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                leave=False,
                disable=not config.get("use_tqdm", True),  # optional flag in config
            )

        for batch in pbar:

            x = batch["observations"].to(device)  # [B,T,C,H,W] in [-1,1]
            B = x.size(0)

            optimizer.zero_grad()

            # Base reconstruction loss
            token_ids, quantized, _, vq_loss = tokenizer.encode(x)
            # Decode with tanh -> [-1, 1]
            recon = tokenizer.decode(quantized, use_tanh=True)

            target_flat = rearrange(x,    "b t c h w -> (b t) c h w")  # [-1,1]
            recon_flat  = rearrange(recon,"b t c h w -> (b t) c h w")  # [-1,1]

            # For LPIPS: stay in [-1,1]
            lpips_val = lpips_fn(recon_flat, target_flat).mean()

            # For pixel recon: go to [0,1]
            target_01 = ((target_flat + 1.0) / 2.0).clamp(0, 1)
            recon_01  = ((recon_flat  + 1.0) / 2.0).clamp(0, 1)

            recon_loss = F.l1_loss(recon_01, target_01)

            loss = recon_loss + vq_loss + config["lpips_weight"] * lpips_val

            # Optional contrastive loss with masking
            cl_loss = torch.tensor(0.0, device=device)
            if contrastive_weight > 0.0 and mask_prob > 0.0:
                x_masked = random_block_mask(x, mask_prob=mask_prob)
                z1 = extract_video_repr(tokenizer, x)
                z2 = extract_video_repr(tokenizer, x_masked)
                cl_loss = info_nce_loss(z1, z2)
                loss = loss + contrastive_weight * cl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), config.get("grad_clip", 1.0))
            optimizer.step()

            running_recon += recon_loss.item() * B
            running_vq += vq_loss.item() * B
            running_cl += cl_loss.item() * B
            count += B
            global_step += 1
            # --- update pbar with current averages ---
            avg_recon = running_recon / max(count, 1)
            avg_vq    = running_vq / max(count, 1)
            avg_cl    = running_cl / max(count, 1)

            pbar.set_postfix(
                recon=f"{avg_recon:.4f}",
                vq=f"{avg_vq:.4f}",
                cl=f"{avg_cl:.4f}",
                lpips=f"{lpips_val.item():.4f}",
                loss=f"{loss.item():.4f}",
            )

        epoch_recon = running_recon / count
        epoch_vq = running_vq / count
        epoch_cl = running_cl / max(count, 1)

        if wandb_run is not None and global_step % config["log_every_steps"] == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/recon_loss": epoch_recon,
                    "train/vq_loss": epoch_vq,
                    "train/contrastive_loss": epoch_cl,
                    "train/contrastive_weight": contrastive_weight,
                    "train/mask_prob": mask_prob,
                },
                step=epoch,
            )

        if writer is not None and global_step % config["log_every_steps"] == 0:
            writer.add_scalar("train/recon_loss", epoch_recon, global_step)
            writer.add_scalar("train/vq_loss", epoch_vq, global_step)
            writer.add_scalar("train/contrastive_loss", epoch_cl, global_step)
            writer.add_scalar("train/contrastive_weight", contrastive_weight, global_step)
            writer.add_scalar("train/mask_prob", mask_prob, global_step)


        print(
            f"[epoch {epoch}] recon={epoch_recon:.4f}  "
            f"vq={epoch_vq:.4f}  cl={epoch_cl:.4f}"
        )

        # Save recon grid every few epochs
        if epoch % config.get("recon_every", 5) == 0:
            save_reconstruction_grid(
                tokenizer,
                val_loader,
                device,
                out_dir,
                epoch,
                wandb_run=wandb_run,
                writer=writer,
            )

        # Evaluate (SSIM, PSNR, FID)
        if epoch % config.get("eval_every", 5) == 0:
            metrics = evaluate_tokenizer(
                tokenizer,
                val_loader,
                device,
                max_batches=config.get("eval_batches", 20),
                max_fid_images=config.get("max_fid_images", 2048),
                inception=inception,
            )
            print(
                f"[eval @ epoch {epoch}] "
                f"MSE={metrics['mse']:.5f}, "
                f"PSNR={metrics['psnr']:.2f} dB, "
                f"SSIM={metrics['ssim']:.4f}, "
                f"FID={metrics['fid']:.2f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "val/mse": metrics["mse"],
                        "val/psnr": metrics["psnr"],
                        "val/ssim": metrics["ssim"],
                        "val/fid": metrics["fid"],
                    },
                    step=epoch,
                )
            if writer is not None:
                writer.add_scalar("val/mse", metrics["mse"], epoch)
                writer.add_scalar("val/psnr", metrics["psnr"], epoch)
                writer.add_scalar("val/ssim", metrics["ssim"], epoch)
                writer.add_scalar("val/fid", metrics["fid"], epoch)

        # Checkpoint
        if epoch % config.get("ckpt_every", 10) == 0:
            ckpt_path = os.path.join(out_dir, f"vqpt_epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": tokenizer.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            print(f"[ckpt] saved {ckpt_path}")
    # after the training loop
    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    # Minimal example config – tweak as needed
    cfg = {
        "data_dir": PARENT_DIR / "transition_data",  # root containing dmc_vb/humanoid_walk/...
        "domain_name": "humanoid",
        "task_name": "walk",
        "sequence_length": 1,   # 1 frame per "video" for VQ pretraining
        "frame_stack": 1,
        "in_channels": 3,
        "code_dim": 256,
        "num_codes": 8192,
        "downsample": 4,
        "base_channels": 128,
        "kmeans_init": True,
        "dropout": 0.1, 
        "commitment_use_cross_entropy_loss": False,
        "orthogonal_reg_weight": 0.0,
        "orthogonal_reg_active_codes_only": True,
        "orthogonal_reg_max_codes": 512,
        "codebook_diversity_loss_weight": 0.0,
        "codebook_diversity_temperature": 0.1,
        "threshold_ema_dead_code": 0,
        "img_height": 64,
        "img_width": 64,
        "weight_decay": 1e-5,
        "batch_size": 45,
        "num_workers": 4,
        "epochs": 80,
        "lpips_weight": 1.0,
        "lr": 2e-4,
        "grad_clip": 1.0,
        "log_every_steps": 20,
        "commitment_weight": 0.25,
        "contrastive_weight": 0.01,  # set to 0.0 to disable CL
        "mask_prob": 0.4,
        "out_dir": PARENT_DIR / "results" / "vqpt_pretrain_dmc_vb",
        "recon_every": 5,
        "eval_every": 5,
        "ckpt_every": 10,
        "eval_batches": 20,
        "max_fid_images": 1024,
        "num_quantizers": 1,
        "use_wandb": False,          # set False to turn off wandb
        "use_tensorboard": True,   # set True to use TensorBoard instead
        # wandb
        'wandb_project': 'dpgmm-vrnn-dmc',
        'wandb_entity': 'zsheikhb',        
        "wandb_run_name": "vqpt_humanoid_walk_pretrain",

    }
    train_vqpt_tokenizer_dmc_vb(cfg)
