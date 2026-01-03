#!/usr/bin/env python3
"""
Visualize Canny edge maps (raw + thresholded) on DMC VB frames, and show how the
edge guide looks after resizing to VDVAE hierarchy resolutions (64/32/16/8/4)
using the same _prep_edge logic you use in vae.py.

Saves a PNG under: <parent_of_transition_data>/results/canny_viz/

Example:
  python -m VRNN.viz_canny_edges \
    --data_dir /scratch/memole/AIME/transition_data \
    --domain humanoid --task walk --policy_level medium --split train \
    --sequence_length 10 --frame_stack 4 --idx 0 --t 0 \
    --k_gaussian 5 --sigma 1.5 \
    --auto_q_low 0.90 --auto_q_high 0.95 --hysteresis \
    --downsample_small area \
    --closing --closing_k 3
"""

# ---- MUST BE FIRST: avoid BLIS/OpenMP thread aborts ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from VRNN.dmc_vb_transition_dynamics_trainer import DMCVBDataset
from VRNN.utils.canny_net import CannyFilter


def denorm_to_01(x_m11: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return (x_m11.clamp(-1, 1) + 1.0) * 0.5


def morph_close(x01: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Torch version of skimage.morphology.binary_closing:
      closing = erosion(dilation(x))
    x01: [B,1,H,W] float in {0,1} (or thresholded before calling)
    """
    pad = k // 2
    dil = F.max_pool2d(x01, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-dil, kernel_size=k, stride=1, padding=pad)
    return ero


class EdgePrep:
    """
    Wrapper that contains your exact _prep_edge logic.
    You can choose what happens for small-resolution downsampling:
      - area: anti-aliased average downsample (your current default)
      - maxpool: "edge presence" downsample (keeps any edge in a cell)
    """
    def __init__(self, edge_channels: int = 1, use_edge_conditioning: bool = True, downsample_small: str = "area"):
        self.edge_channels = edge_channels
        self.use_edge_conditioning = use_edge_conditioning
        assert downsample_small in ["area", "maxpool"]
        self.downsample_small = downsample_small

    def _prep_edge(self, edge_guide: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        """edge_guide: [B,Ce,H0,W0] or [B,H0,W0,Ce] -> edge_map [B,Ce,H,W] matching x spatial size."""
        if not self.use_edge_conditioning:
            return None

        B, _, H, W = x.shape  # x is [B,C,H,W]
        Ce = self.edge_channels

        if edge_guide is None:
            # t=0 / reset: keep channel dims consistent by using zeros
            return torch.zeros(B, Ce, H, W, device=x.device, dtype=x.dtype)

        if edge_guide.dim() != 4:
            raise ValueError(f"edge_guide must be 4D, got {tuple(edge_guide.shape)}")

        # Accept NCHW or NHWC
        if edge_guide.shape[1] == Ce:
            edge = edge_guide
        elif edge_guide.shape[-1] == Ce:
            edge = edge_guide.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"edge_guide channel mismatch: expected Ce={Ce}, got {tuple(edge_guide.shape)}"
            )

        edge = edge.to(device=x.device, dtype=x.dtype)

        src_h, src_w = edge.shape[2], edge.shape[3]
        tgt_h, tgt_w = H, W

        if (src_h, src_w) != (tgt_h, tgt_w):
            # define block resolution r (your blocks are typically square)
            r = min(tgt_h, tgt_w)

            # only treat as downsampling if both dims shrink (or stay)
            downsample = (tgt_h <= src_h) and (tgt_w <= src_w)

            if downsample and r < 32:
                if self.downsample_small == "area":
                    # anti-aliased downsample (keeps thin edges from disappearing as often)
                    edge = F.interpolate(edge, size=(tgt_h, tgt_w), mode="area")
                else:
                    # edge presence: keep any edge in each cell
                    edge = F.adaptive_max_pool2d(edge, output_size=(tgt_h, tgt_w))
            else:
                # keep edges crisp for higher resolutions, or when upsampling
                edge = F.interpolate(edge, size=(tgt_h, tgt_w), mode="nearest")

        return edge


def pick_results_dir(data_dir: Path) -> Path:
    """
    Save into results folder next to transition_data.
    If data_dir ends with transition_data -> parent/results/canny_viz
    Else -> data_dir.parent/results/canny_viz
    """
    data_dir = data_dir.resolve()
    base = data_dir.parent if data_dir.name == "transition_data" else data_dir.parent
    out = base / "results" / "canny_viz"
    out.mkdir(parents=True, exist_ok=True)
    return out


def imshow_edge(ax, e: torch.Tensor, title: str, vmax_quantile: float = 0.995):
    """
    RAW edges are often tiny (e.g. 0..0.02). If you plot with vmax=1,
    everything looks black. So we use a percentile-based vmax.
    """
    e_cpu = e.detach().cpu()
    flat = e_cpu.flatten()
    if flat.numel() == 0:
        vmax = 1.0
    else:
        vmax = float(torch.quantile(flat, torch.tensor(vmax_quantile)))
        if vmax <= 1e-12:
            vmax = float(flat.max()) if float(flat.max()) > 0 else 1.0

    ax.imshow(e_cpu.numpy(), cmap="gray", vmin=0.0, vmax=vmax)
    ax.set_title(f"{title}\nvmax(p{vmax_quantile*100:.1f})={vmax:.3g}")
    ax.axis("off")


def build_dataset(args, seq_len: int):
    return DMCVBDataset(
        data_dir=str(args.data_dir),
        domain_name=args.domain,
        task_name=args.task,
        policy_level=args.policy_level,
        split=args.split,
        sequence_length=seq_len,
        frame_stack=args.frame_stack,
        img_height=args.base_res,
        img_width=args.base_res,
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True,
                    help="Path that contains dmc_vb/ (usually .../transition_data).")
    ap.add_argument("--domain", type=str, default="humanoid")
    ap.add_argument("--task", type=str, default="walk")
    ap.add_argument("--policy_level", type=str, default="medium", choices=["expert", "medium", "mixed", "random"])
    ap.add_argument("--split", type=str, default="train", choices=["train", "eval", "val", "test"])
    ap.add_argument("--frame_stack", type=int, default=4)
    ap.add_argument("--sequence_length", type=int, default=10)

    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--base_res", type=int, default=64)

    # CannyFilter parameters (your differentiable canny)
    ap.add_argument("--k_gaussian", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--k_sobel", type=int, default=3)

    # Thresholding options
    ap.add_argument("--low", type=float, default=None, help="manual low threshold for thin_edges scale")
    ap.add_argument("--high", type=float, default=None, help="manual high threshold for thin_edges scale")
    ap.add_argument("--hysteresis", action="store_true", help="use soft hysteresis inside CannyFilter")

    # Auto thresholds from quantiles of RAW thin_edges (recommended)
    ap.add_argument("--auto_q_low", type=float, default=0.90,
                    help="quantile for low threshold (e.g. 0.90)")
    ap.add_argument("--auto_q_high", type=float, default=0.95,
                    help="quantile for high threshold (e.g. 0.95)")
    ap.add_argument("--no_auto", action="store_true",
                    help="disable auto thresholds (only raw edge shown unless --low is provided)")

    # Use your exact _prep_edge behavior; choose what happens for small r downsample
    ap.add_argument("--downsample_small", type=str, default="area", choices=["area", "maxpool"],
                    help="When downsampling to r<32: 'area' (your default) or 'maxpool' (edge presence).")

    # Morphology
    ap.add_argument("--closing", action="store_true")
    ap.add_argument("--edge_thr", type=float, default=0.5, help="binarize threshold before closing")
    ap.add_argument("--closing_k", type=int, default=3)

    # Plotting knobs
    ap.add_argument("--raw_vmax_q", type=float, default=0.995,
                    help="vmax quantile for raw edge display (0.99..0.999)")

    args = ap.parse_args()

    args.data_dir = Path(args.data_dir)
    device = torch.device("cpu")

    # Also force torch threading to 1 (helps with BLIS/OpenMP weirdness)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # ---- Build dataset with fallback if seq_len too long ----
    seq_try = [args.sequence_length, 16, 12, 10, 8, 6, 4, 2, 1]
    seq_try = [s for i, s in enumerate(seq_try) if s not in seq_try[:i]]

    ds = None
    used_seq = None
    last_err = None
    for s in seq_try:
        try:
            ds = build_dataset(args, s)
            used_seq = s
            break
        except RuntimeError as e:
            last_err = e
            if "No valid episodes long enough" in str(e):
                continue
            raise

    if ds is None:
        raise last_err

    if len(ds) == 0:
        raise RuntimeError("Dataset length is 0 after filtering; check domain/task/policy_level/split.")

    idx = max(0, min(args.idx, len(ds) - 1))
    sample = ds[idx]

    obs = sample["observations"]  # [T, 3*frame_stack, H, W] in [-1,1]
    T, Cfs, H, W = obs.shape
    t = max(0, min(args.t, T - 1))

    x = obs[t]                           # [3*fs, H, W]
    x_rgb = x[-3:, :, :].unsqueeze(0)    # [1,3,H,W], most recent RGB in stack

    if x_rgb.shape[-2:] != (args.base_res, args.base_res):
        x_rgb = F.interpolate(x_rgb, size=(args.base_res, args.base_res),
                              mode="bilinear", align_corners=False)

    x01 = denorm_to_01(x_rgb)  # [1,3,base,base] in [0,1]

    # ---- Canny ----
    canny = CannyFilter(
        k_gaussian=args.k_gaussian,
        sigma=args.sigma,
        k_sobel=args.k_sobel,
        use_cuda=False,
    ).to(device)

    # RAW thin edges
    edge_raw = canny(x01, low_threshold=None, high_threshold=None, hysteresis=False)  # [1,1,H,W]

    flat = edge_raw.flatten()
    qs = torch.quantile(flat, torch.tensor([0.50, 0.90, 0.95, 0.99]))
    print("RAW thin_edges stats:",
          f"min={float(flat.min()):.6f}",
          f"max={float(flat.max()):.6f}",
          f"q50={float(qs[0]):.6f}",
          f"q90={float(qs[1]):.6f}",
          f"q95={float(qs[2]):.6f}",
          f"q99={float(qs[3]):.6f}")

    # Decide thresholding
    edge_thr = None
    low = args.low
    high = args.high

    if not args.no_auto and low is None:
        # auto thresholds from quantiles (recommended)
        low = float(torch.quantile(flat, torch.tensor(args.auto_q_low)))
        high = float(torch.quantile(flat, torch.tensor(args.auto_q_high)))
        print(f"Auto thresholds: low=q{args.auto_q_low:.2f}={low:.6f}, high=q{args.auto_q_high:.2f}={high:.6f}")

    if low is not None:
        edge_thr = canny(x01, low_threshold=low, high_threshold=high, hysteresis=args.hysteresis)
        # edge_thr is meant to be ~[0,1] when thresholds are active

    # ---- Resize across hierarchy using your _prep_edge ----
    prep = EdgePrep(edge_channels=1, use_edge_conditioning=True, downsample_small=args.downsample_small)

    res_levels = [64, 32, 16, 8, 4]
    res_levels = [r for r in res_levels if r <= args.base_res]

    def resize_with_prep(edge_map: torch.Tensor, r: int) -> torch.Tensor:
        # x dummy only used for target H,W and dtype/device
        x_dummy = torch.empty(edge_map.shape[0], 1, r, r, device=edge_map.device, dtype=edge_map.dtype)
        e_r = prep._prep_edge(edge_map, x_dummy)  # [B,1,r,r]
        return e_r

    raw_up = []
    thr_up = []
    close_up = []

    for r in res_levels:
        e_r = resize_with_prep(edge_raw, r)
        e_up = F.interpolate(e_r, size=(args.base_res, args.base_res), mode="nearest")
        raw_up.append(e_up.squeeze(0).squeeze(0))

        if edge_thr is not None:
            t_r = resize_with_prep(edge_thr, r)
            t_up = F.interpolate(t_r, size=(args.base_res, args.base_res), mode="nearest")
            thr_up.append(t_up.squeeze(0).squeeze(0))

            if args.closing:
                t_bin = (t_r > args.edge_thr).float()
                t_close = morph_close(t_bin, k=args.closing_k)
                t_close_up = F.interpolate(t_close, size=(args.base_res, args.base_res), mode="nearest")
                close_up.append(t_close_up.squeeze(0).squeeze(0))

    # ---- Plot grid ----
    # Row 0: RGB + raw edges
    # Row 1: thresholded (if available)
    # Row 2: closing (if requested)
    nrows = 1 + (1 if edge_thr is not None else 0) + (1 if (args.closing and edge_thr is not None) else 0)
    ncols = 1 + len(res_levels)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    # RGB
    img_np = x01.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"RGB (idx={idx}, t={t}, seq={used_seq}, fs={args.frame_stack})")
    axes[0, 0].axis("off")

    # RAW edges at resolutions
    for j, r in enumerate(res_levels, start=1):
        imshow_edge(axes[0, j], raw_up[j - 1], f"RAW edge @ {r} → {args.base_res}", vmax_quantile=args.raw_vmax_q)

    row = 1
    if edge_thr is not None:
        axes[row, 0].text(
            0.5, 0.5,
            f"THRESHOLDED\nlow={low:.4g}\nhigh={high:.4g}\nhyst={int(args.hysteresis)}\nsmall={args.downsample_small}",
            ha="center", va="center", fontsize=12
        )
        axes[row, 0].axis("off")

        for j, r in enumerate(res_levels, start=1):
            axes[row, j].imshow(thr_up[j - 1].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, j].set_title(f"THR @ {r} → {args.base_res}")
            axes[row, j].axis("off")
        row += 1

        if args.closing:
            axes[row, 0].text(
                0.5, 0.5,
                f"CLOSING\n(bin>{args.edge_thr})\nk={args.closing_k}",
                ha="center", va="center", fontsize=12
            )
            axes[row, 0].axis("off")
            for j, r in enumerate(res_levels, start=1):
                axes[row, j].imshow(close_up[j - 1].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                axes[row, j].set_title(f"CLOSE @ {r} → {args.base_res}")
                axes[row, j].axis("off")

    plt.tight_layout()

    # ---- Save ----
    out_dir = pick_results_dir(args.data_dir)
    thr_tag = "no_thr" if edge_thr is None else f"low{low:.3g}_high{high:.3g}_hyst{int(args.hysteresis)}"
    fname = (
        f"canny_{args.domain}_{args.task}_{args.policy_level}_{args.split}"
        f"_idx{idx}_t{t}_seq{used_seq}_fs{args.frame_stack}"
        f"_res{args.base_res}_kg{args.k_gaussian}_sig{args.sigma}_ks{args.k_sobel}"
        f"_{thr_tag}_small{args.downsample_small}"
        f"{'_closing' if args.closing else ''}.png"
    )
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[saved]", str(save_path))


if __name__ == "__main__":
    main()
