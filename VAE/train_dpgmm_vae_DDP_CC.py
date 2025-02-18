import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
import glob
import logging
from dpgmm_stickbreaking_prior_vae import DPGMMVariationalAutoencoder
from models import AverageMeter
from collections import defaultdict
from torchvision import transforms
import time
import torch.nn.functional as F
import scipy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix as contingency_matrix

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Configure logging
# format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='stick-breaking DPGMM VAE')
parser.add_argument('--max_components', type=int,
                    default=38, help='Max # of GMM components')
parser.add_argument('--latent_dim', type=int, default=40,
                    help='Latent dimension size')
parser.add_argument('--hidden_dim', type=int, default=35,
                    help='Hidden dimension size')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Beta hyperparameter')
parser.add_argument('--lambda_img', type=float,
                    default=1.0, help='Lambda for image loss')
parser.add_argument('--lambda_latent', type=float,
                    default=3.0, help='Lambda for latent loss')
parser.add_argument('--n_critic', type=int, default=3,
                    help='Number of critic steps')
parser.add_argument('--grad_clip', type=float, default=0.5,
                    help='Gradient clipping threshold')
parser.add_argument('--num_epochs', type=int,
                    default=901, help='Number of epochs')
parser.add_argument('--img_disc_channels', type=int,
                    default=16, help='Image discriminator channels')
parser.add_argument('--img_disc_layers', type=int,
                    default=5, help='Image discriminator layers')
parser.add_argument('--latent_disc_layers', type=int,
                    default=3, help='Latent discriminator layers')
parser.add_argument('--use_actnorm', type=bool, default=True,
                    help='Use activation normalization')

parser.add_argument('--result_dir', type=str, default='results',
                    help='Directory results saved to')

parser.add_argument(
    '--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


class FIDScore:
    def __init__(self, device):
        self.device = device
        # Initialize Inception network properly with modern practices
        self.inception = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=True
        ).to(device)
        # Remove final classification layer and set to features only
        self.inception.aux_logits = False
        self.inception.fc = nn.Identity()
        self.inception.eval()

    def preprocess_images(self, images):
        """
        Preprocess images to match Inception-v3 expectations.
        Args:
            images: torch.Tensor of shape (N, 3, H, W) with values in [-1, 1]
        Returns:
            preprocessed images: torch.Tensor of shape (N, 3, 299, 299)
        """
        # Convert from [-1, 1] to [0, 255]
        images = (images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

        # Resize to Inception input size
        images = F.interpolate(images.float(), size=(299, 299))
        # Apply ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = normalize(images)

        return images

    def get_features(self, images):
        """
        Extract features using Inception-v3 network.
        Args:
            images: torch.Tensor of shape (N, 3, H, W) in range [-1, 1]
        Returns:
            features: torch.Tensor of shape (N, 2048)
        """
        if not isinstance(images, torch.Tensor):
            raise ValueError("Input must be a torch tensor")

        if len(images.shape) != 4:
            raise ValueError(
                f"Input must be a 4D tensor (B,C,H,W), got shape {images.shape}")

        with torch.no_grad():
            # Preprocess images
            preprocessed_images = self.preprocess_images(images)
            # Extract features
            features = self.inception(preprocessed_images.to(self.device))

        return features

    def calculate_statistics(self, features):
        """
        Calculate mean and covariance statistics of features.
        Args:
            features: torch.Tensor of shape (N, 2048)
        Returns:
            mu: mean features
            sigma: covariance matrix
        """
        features = features.cpu().numpy()
        mu = np.mean(features, axis=0)

        # Compute covariance
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate Frechet distance between two sets of statistics.
        Args:
            mu1, mu2: mean features for each distribution
            sigma1, sigma2: covariance matrices for each distribution
        Returns:
            fid: Frechet distance score
        """
        # Ensure covariance matrices are positive semi-definite
        eps = np.finfo(np.float32).eps
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps

        # Compute mean difference
        diff = mu1 - mu2
        mean_diff_sq = np.sum(diff**2)

        # Compute sqrtm(sigma1 @ sigma2) with improved numerical stability
        try:
            covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real  # Take only real part if imaginary component is negligible
        except ValueError:
            # SVD-based fallback for sqrtm computation
            U, S, Vt = np.linalg.svd(sigma1 @ sigma2)
            covmean = U @ np.diag(np.sqrt(S)) @ Vt

        # Compute final FID score
        trace_covmean = np.trace(covmean)
        fid_score = mean_diff_sq + \
            np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean

        return float(fid_score)

    def calculate_fid(self, real_images, fake_images):
        """
        Calculate FID score between real and fake images.
        Args:
            real_images: torch.Tensor of shape (N, 3, H, W) in range [-1, 1]
            fake_images: torch.Tensor of shape (N, 3, H, W) in range [-1, 1]
        Returns:
            fid_score: float, the FID score between the two sets of images
        """
        try:
            # Move images to correct device
            real_images = real_images.to(self.device)
            fake_images = fake_images.to(self.device)

            # Extract features
            real_features = self.get_features(real_images)
            fake_features = self.get_features(fake_images)

            # Calculate statistics
            mu1, sigma1 = self.calculate_statistics(real_features)
            mu2, sigma2 = self.calculate_statistics(fake_features)

            # Calculate FID score
            fid_score = self.calculate_frechet_distance(
                mu1, sigma1, mu2, sigma2)

            return fid_score

        except Exception as e:
            print(f"Error calculating FID: {str(e)}")
            return float('inf')


def compute_metrics(model, test_loader, device, writer, global_step, epoch=None, max_samples=100, visualize_tsne=False, result_dir=None, label_name_map=None):
    """Compute clustering metrics and image quality metrics."""
    model.eval()

    # Metric accumulators
    acc_meter = AverageMeter()
    fid = FIDScore(device=device)
    total_loss = 0
    all_predictions, all_labels, all_latents = [], [], []
    all_real_images, all_fake_images = [], []

    with torch.no_grad():
        for batch_idx, (data, masks, labels) in enumerate(test_loader):
            if len(all_real_images) >= max_samples:
                break  # Limit samples for efficiency

            data = data.to(device)
            data = (data - 0.5) * 2.0  # Normalize input data

            # Forward pass
            reconstruction, params = model(data)
            cluster_probs = params['prior_dist'].mixture_distribution.probs
            latent_vectors = params['z'].detach()

            # Store predictions & labels
            all_predictions.append(cluster_probs)
            all_latents.append(latent_vectors.cpu().numpy())
            all_labels.append(labels.to(device))
            all_real_images.append(data)
            all_fake_images.append(reconstruction)

            # Compute loss
            losses = model.module.compute_loss(data, reconstruction, params)
            total_loss += losses['loss'].item()

            # Compute accuracy
            cluster_assignments = cluster_probs.argmax(1)
            acc = (cluster_assignments == labels.to(device)).float().mean()
            acc_meter.update(acc.item(), data.size(0))

    # Convert collected data
    all_predictions = torch.cat(all_predictions, dim=0)
    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute clustering metrics
    metrics = {}
    # Add learning rates
    metrics['gen_lr'] = model.module.gen_scheduler.get_last_lr()[0]
    metrics['img_disc_lr'] = model.module.img_disc_scheduler.get_last_lr()[0]
    metrics['latent_disc_lr'] = model.module.latent_disc_scheduler.get_last_lr()[
        0]

    cluster_assignments = all_predictions.argmax(1).cpu().numpy()
    true_labels = all_labels.cpu().numpy()
    # Normalized Mutual Information
    metrics['nmi'] = normalized_mutual_info_score(true_labels,
                                                  cluster_assignments,
                                                  average_method='arithmetic'  # Can also be 'geometric' or 'min'
                                                  )
    # Measures similarity between two clusterings and it is adjusted for chance (can be negative if worse than random)
    metrics['ari'] = adjusted_rand_score(true_labels, cluster_assignments)

    # Compute cluster purity
    def compute_purity(predictions, labels):
        # Simpler metric that measures percentage of samples correctly clustered
        contingency = contingency_matrix(labels, predictions)

        # For each cluster, find the most common true label
        # Sum these and divide by total samples
        return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

    metrics['purity'] = compute_purity(cluster_assignments, true_labels)

    # Compute effective number of clusters
    metrics['effective_clusters'] = model.module.prior.get_effective_components(
        all_predictions.mean(0)).item()

    # Compute entropy of cluster assignments
    cluster_entropy = -(all_predictions *
                        torch.log(all_predictions + 1e-10)).sum(1).mean()
    metrics['cluster_entropy'] = cluster_entropy.item()

    # Compute FID Score
    try:
        real_images = torch.cat(all_real_images, dim=0)[:max_samples]
        fake_images = torch.cat(all_fake_images, dim=0)[:max_samples]
        metrics['fid_score'] = fid.calculate_fid(real_images, fake_images)
    except Exception as e:
        print(f"Error in FID calculation: {str(e)}")
        metrics['fid_score'] = float('inf')

    # Log metrics
    writer.add_scalar('test/accuracy', acc_meter.avg, global_step)
    for name, value in metrics.items():
        writer.add_scalar(f'test/{name}', value, global_step)
    writer.add_scalar('test/loss', total_loss / len(test_loader), global_step)

    # visualize TSNE
    if visualize_tsne:
        ImagePlotUtils.plot_tsne_embeddings(
            all_latents,
            all_labels.cpu().numpy(),
            epoch,
            f"{result_dir}/tsne",
            label_name_map=label_name_map
        )

    return metrics, acc_meter.avg, total_loss / len(test_loader)


class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
    ):
        # Always include both class and segmentation targets
        if isinstance(target_types, str):
            target_types = [target_types, "category"]
        elif "category" not in target_types:
            target_types = list(target_types) + ["category"]

        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
        )
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        input, targets = super().__getitem__(idx)
        segmentation_mask, class_label = targets

        if not isinstance(input, torch.Tensor):
            input = self.transform(input)
        if not isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = self.mask_transform(segmentation_mask)

        mask = (segmentation_mask > 0).float()
        mask_onehot = torch.zeros((2, mask.shape[1], mask.shape[2]))
        mask_onehot[0] = 1.0 - mask.squeeze(0)  # background
        mask_onehot[1] = mask.squeeze(0)        # foreground

        return input, mask_onehot, class_label


class MNISTAugmented(torchvision.datasets.MNIST):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        target_types="segmentation",
        download=False,
    ):
        # Ensure both "category" and "segmentation" are included
        if isinstance(target_types, str):
            target_types = [target_types, "category"]
        elif "category" not in target_types:
            target_types = list(target_types) + ["category"]

        # Initialize the parent MNIST class
        super().__init__(
            root=root,
            train=(split == 'train'),  # Split logic
            transform=None,  # We handle transforms manually
            target_transform=None,
            download=download,
        )

        # Image transformations (resize, normalize, etc.)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 128x128
            # Convert MNIST to 3-channel (RGB) images
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[
                                 0.3081, 0.3081, 0.3081]),  # Normalize as RGB channels
        ])

        # Mask transformations (resize, convert to tensor, etc.)
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize masks to 128x128
            transforms.ToTensor(),  # Convert masks to tensors
        ])

    def __getitem__(self, idx):
        # Retrieve the image and label
        input, label = super().__getitem__(idx)

        # Apply image transformation (resize, convert to RGB, normalize, etc.)
        if not isinstance(input, torch.Tensor):
            input = self.transform(input)

        # For segmentation, we assume masks are available as part of target types
        # Assuming that we have a segmentation mask, which in practice would need to be constructed
        segmentation_mask = self.create_segmentation_mask(label)

        # Apply mask transformations (resize, convert to tensor)
        if not isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = self.mask_transform(segmentation_mask)

        # Generate one-hot encoded mask (foreground and background)
        mask = (segmentation_mask > 0).float()
        # Two channels for background and foreground
        mask_onehot = torch.zeros((2, mask.shape[1], mask.shape[2]))
        mask_onehot[0] = 1.0 - mask.squeeze(0)  # Background
        mask_onehot[1] = mask.squeeze(0)        # Foreground

        # Return the image, one-hot encoded mask, and class label
        return input, mask_onehot, label

    def create_segmentation_mask(self, label):
        # Create a dummy segmentation mask as a simple example
        # Here we simulate a segmentation mask where the foreground is the object in the center
        mask = torch.zeros((1, 28, 28))  # MNIST images are 28x28
        # Simply put the foreground in the center (this can be more complex in practice)
        mask[:, 10:18, 10:18] = 1  # Simulate a square foreground mask
        return mask


class ImagePlotUtils:
    @staticmethod
    def save_comparison_grid(original_images, reconstructed_images, filename, nrow=None):
        if nrow is None:
            nrow = original_images.size(0)

        # Create grid tensor
        b, c, h, w = original_images.shape
        grid = torch.zeros(size=(b, 2, c, h, w))
        grid[:, 0] = original_images.detach().cpu()
        grid[:, 1] = reconstructed_images.detach().cpu()
        grid = grid.view(-1, c, h, w)

        # Create comparison grid
        comparison = torchvision.utils.make_grid(
            grid, nrow=nrow, normalize=True, value_range=(-1, 1)
        )

        # Save image
        comparison = comparison.numpy().transpose((1, 2, 0))
        plt.figure(figsize=(15, 5))
        plt.imshow(comparison)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def plot_tsne_embeddings(embeddings, labels, epoch, filename_prefix, label_name_map=None):
        plt.figure(figsize=(10, 10))

        # Adjust perplexity based on sample size
        n_samples = embeddings.shape[0]
        # Ensure perplexity is less than n_samples
        perplexity = min(30, n_samples - 1)

        # Create and fit t-SNE
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            max_iter=1000,  # Changed from n_iter
            learning_rate='auto'
        )
        try:
            z_embedded = tsne.fit_transform(embeddings)
            unique_labels = np.unique(labels)

            # we need lots of colors
            cmap = plt.get_cmap("gist_ncar")
            num_classes = len(unique_labels)

            for i, label in enumerate(unique_labels):
                mask = labels == label

                if label_name_map:
                    label = label_name_map[label]

                plt.scatter(
                    z_embedded[mask, 0],
                    z_embedded[mask, 1],
                    s=10,
                    c=[cmap(i / num_classes)],
                    alpha=0.6,
                    label=f'Class {label}'
                )

            plt.title(f"Latent Space t-SNE (Epoch {epoch})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_{epoch}.png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error in t-SNE visualization: {str(e)}")


def setup(dist_backend, init_method, world_size):
    # assert torch.distributed.is_available()
    # print("PyTorch Distributed available.")
    # print("  Backends:")
    # print(f"    Gloo: {torch.distributed.is_gloo_available()}")
    # print(f"    NCCL: {torch.distributed.is_nccl_available()}")
    # print(f"    MPI:  {torch.distributed.is_mpi_available()}")

    ngpus_per_node = torch.cuda.device_count()
    print(f"  Found {ngpus_per_node} GPUs per node.")

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

    print(f"    Local Rank {local_rank} and rank {rank}")

    current_device = local_rank

    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    # init the process group
    init_process_group(backend=dist_backend,
                       init_method=init_method, world_size=world_size, rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))

    return current_device, rank, world_size


def main():
    print("Starting...")

    args = parser.parse_args()

    # Hyperparameters
    # config = {
    #     'max_components': 38,
    #     'latent_dim': 40,
    #     'hidden_dim': 35,
    #     'batch_size': 128,
    #     'lr': 1e-4,
    #     'beta': 1.0,
    #     'lambda_img': 1.0,
    #     'lambda_latent': 3.0,
    #     'n_critic': 3,
    #     'grad_clip': 0.5,
    #     'num_epochs': 1801,
    #     'img_disc_channels': 16,
    #     'img_disc_layers': 5,
    #     'latent_disc_layers': 3,
    #     'use_actnorm': True
    # }
    config = vars(args)

    # Setup device and paths
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the process group
    current_device, rank, world_size = setup(
        dist_backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size)
    is_master = rank == 0
    device = torch.device("cuda", rank % torch.cuda.device_count())
    print('device being used:', device)

    result_dir = Path(config['result_dir'])
    result_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir='runs/dpgmm_vae_experiment')

    print('From Rank: {}, ==> Preparing data..'.format(rank))
    # Load dataset
    train_dataset = OxfordIIITPetsAugmented(
        root='~/scratch/datasets',
        split="trainval",
        download=False
    )

    test_dataset = OxfordIIITPetsAugmented(
        root='~/scratch/datasets',
        split='test',
        download=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=DistributedSampler(train_dataset, shuffle=True)

        # shuffle=True,
        # num_workers=4,  # Reduced from 8
        # pin_memory=True,
        # prefetch_factor=2,
        # persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=DistributedSampler(test_dataset)

        # shuffle=False,
        # num_workers=4, # Reduced from 8
        # pin_memory=True,
        # prefetch_factor=2
    )

    # Initialize model
    input_dim = next(iter(train_loader))[0].shape[-1]

    model = DPGMMVariationalAutoencoder(
        max_components=config['max_components'],
        input_dim=input_dim,
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        img_disc_channels=config['img_disc_channels'],
        img_disc_layers=config['img_disc_layers'],
        latent_disc_layers=config['latent_disc_layers'],
        device=device,
        use_actnorm=config['use_actnorm'],
        learning_rate=config['lr']  # Add this parameter
    )

    # DDP on all cuda devices
    model = DDP(model, device_ids=[current_device])

    # Add gradient checkpointing
    model.module.encoder.gradient_checkpointing_enable()
    model.module.decoder.gradient_checkpointing_enable()

    global_step = 0
    # Training loop
    best_fid = float('inf')  # This was missing but used in the code
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_losses = defaultdict(float)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        model.module.update_temperature(epoch)

        # shuffle across epochs
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, (data, masks, labels) in enumerate(train_loader):

            data_time.update(time.time() - end)

            data = data.to(device)
            data = (data - 0.5) * 2.0  # Scale to [-1, 1]

            losses, metrics, latent_vectors = model.module.training_step(
                data,
                beta=config['beta'],
                n_critic=config['n_critic'],
                lambda_img=config['lambda_img'],
                lambda_latent=config['lambda_latent'],
                current_step=global_step
            )
            # Measure elapsed time for batch
            batch_time.update(time.time() - end)
            end = time.time()  # Reset timer
            # update steps
            global_step += 1
            for k, v in metrics.items():
                epoch_losses[f'gen_{k}'] += v

            if epoch % 20 == 0 and batch_idx == 0:  # Run only at epoch intervals
                metrics, accuracy, total_loss = compute_metrics(
                    model,
                    test_loader,
                    device,
                    writer,
                    global_step,
                    epoch,
                    max_samples=1000,
                    visualize_tsne=True,
                    result_dir=result_dir,
                    label_name_map=test_dataset.classes  # for OxfordDataset
                )

                # GPU metrics
                # GPU_mem_alloc = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
                # GPU_mem_reserved = torch.cuda.memory_reserved() / 1024**2    # Convert to MB
                GPU_max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

                # Enhanced logging with all metrics
                logger.info(
                f'\nStep: {global_step} | Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                f'Rank {rank}, Using GPU: {torch.cuda.current_device()}\n'
                # f'Allocated: {GPU_mem_alloc:.2f} MB, Reserved: {GPU_mem_reserved:.2f} MB, Max: {GPU_max_mem:.2f} MB'
                f'Max GPU memory: {GPU_max_mem:.2f} MB'
                f'({100. * batch_idx / len(train_loader):.0f}%)]\n'
                f'Temperature: {model.module.temperature.item():.4f}\n'
                f'Clustering Metrics:\n'
                f'  • Accuracy: {accuracy:.4f}\n'
                f'  • NMI Score: {metrics["nmi"]:.4f}\n'
                f'  • ARI Score: {metrics["ari"]:.4f}\n'
                f'  • Purity: {metrics["purity"]:.4f}\n'
                f'  • Effective Clusters: {metrics["effective_clusters"]:.1f}\n'
                f'  • Cluster Entropy: {metrics["cluster_entropy"]:.4f}\n'
                f'Quality Metrics:\n'
                f'  • FID Score: {metrics["fid_score"]:.4f}\n'
                f'  • Total Loss: {total_loss:.4f}\n'
                f'Learning Rates:\n'
                f'  • Generator: {metrics["gen_lr"]:.2e}\n'
                f'  • Image Disc: {metrics["img_disc_lr"]:.2e}\n'
                f'  • Latent Disc: {metrics["latent_disc_lr"]:.2e}\n'
                f'Training Statistics:\n'
                f'  • Batch Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s avg)\n'
                f'  • Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s avg)'
                )

                # Add a separator for better readability
                logger.info('-' * 80)
                # Save best model based on FID score
                if metrics["fid_score"] < best_fid:
                    best_fid = metrics["fid_score"]
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'gen_scheduler': model.module.gen_scheduler.state_dict(),
                        'img_disc_scheduler': model.module.img_disc_scheduler.state_dict(),
                        'latent_disc_scheduler': model.module.latent_disc_scheduler.state_dict(),
                        'optimizers': {
                            'gen': model.module.gen_optimizer.state_dict(),
                            'img_disc': model.module.img_disc_optimizer.state_dict(),
                            'latent_disc': model.module.latent_disc_optimizer.state_dict()
                        },
                        'metrics': {
                            'fid': metrics["fid_score"],
                            'accuracy': accuracy
                        }
                    }, f"{result_dir}/best_model.pt")

            # Visualization
            if batch_idx == 0 and epoch % 50 == 0:

                with torch.no_grad():
                    # Save reconstructions
                    reconstruction = model.module.decoder(latent_vectors)
                    ImagePlotUtils.save_comparison_grid(
                        data,
                        reconstruction,
                        f"{result_dir}/reconstructions_epoch_{epoch}.png"
                    )

                    # t-SNE visualization

                    # ImagePlotUtils.plot_tsne_embeddings(
                    #     latent_vectors.cpu().numpy(),
                    #     labels.cpu().numpy(),
                    #     epoch,
                    #     f"{result_dir}/tsne"
                    # )

            # Log statistics
            if batch_idx % 500 == 0:

                writer.add_scalar(
                    'lr/generator', metrics['gen_lr'], global_step)
                writer.add_scalar('lr/img_discriminator',
                                  metrics['img_disc_lr'], global_step)
                writer.add_scalar('lr/latent_discriminator',
                                  metrics['latent_disc_lr'], global_step)

                # Get clustering statistics
                stats = model.module.get_cluster_statistics(data)
                writer.add_scalar('stats/active_clusters',
                                  stats['active_clusters'], epoch)
                writer.add_scalar('stats/temperature',
                                  stats['temperature'], epoch)
                writer.add_scalar('stats/mean_entropy',
                                  stats['mean_entropy'], epoch)

        # Average losses for epoch
        for k, v in epoch_losses.items():
            v /= len(train_loader)
            writer.add_scalar(f'loss/{k}', v, epoch)

        # Log final batch losses
        logger.info(
            f'Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
            f'Training Stats:\n'
            f'  - Total Loss: {metrics["total_loss"]:.6f}\n'
            f'  - Generator LR: {metrics["gen_lr"]:.6f}\n'
            f'  - Image Disc. Loss: {metrics["img_disc_loss"]:.6f}\n'
            f'  - Latent Disc. Loss: {metrics["latent_disc_loss"]:.6f}\n'
            f'  - KL Loss: {metrics["kl_z"]:.6f}\n'
            f'  - Kumar-Beta KL: {metrics["kumar_beta_kl"]:.6f}\n'
            f'  - Unused Components: {metrics["unused_components"]:.6f}\n'
            f'  - Effective Components: {metrics["effective_components"]:.2f}\n'
            f'  - Batch Time: {batch_time.val:.3f}s'
        )
        # Log timing metrics at the end of each epoch
        writer.add_scalar('time/avg_batch', batch_time.avg, epoch)
        writer.add_scalar('time/avg_data_loading', data_time.avg, epoch)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
