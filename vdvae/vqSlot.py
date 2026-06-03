import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from geomstats.geometry.hypersphere import Hypersphere
from einops import rearrange, repeat
from torchvision import models
from torch import einsum
import torch.distributed as distributed
from torch.cuda.amp import autocast
import os
import random
from contextlib import contextmanager, ExitStack
import cv2
import scipy
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from pymatting.util.util import row_sum
from pymatting.util.kdtree import knn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.autograd import Variable
import torchvision.models as models
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from einops import rearrange

from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
    return model



def create_histogram(k, img_size, cbidx):
    image = np.zeros((2*k + 2, 2*k + 2, 3), dtype='uint8')

    cbidx = cbidx.detach().cpu().numpy()
    for uidx in np.unique(cbidx):
        count = np.sum(cbidx == uidx)
        image[:int(2*count), int(2*uidx):int(2*uidx+2), :] = (125, 255, 25)

    image = cv2.resize(image, img_size, cv2.INTER_NEAREST)
    image = image*1.0/255
    image = np.flipud(image)
    return 1 - image.transpose(2, 0, 1)


def visualize(image, recon_orig, attns, cbidxs, max_slots, N=8):
    _, _, H, W = image.shape
    attns = attns.permute(0, 1, 4, 2, 3)
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon_orig = recon_orig[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    attns = attns[:N].expand(-1, -1, 3, H, W)

    histograms = np.array([create_histogram(max_slots, (W, H), idxs) for idxs in cbidxs[:N]])
    histograms = torch.from_numpy(histograms).to(image.device).type(image.dtype).unsqueeze(1)

    return torch.cat((image, recon_orig, attns, histograms), dim=1).view(-1, 3, H, W)


def linear_warmup(step, start_value, final_value, start_step, final_step):

    assert start_value <= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b

    return value


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def hsphere_init(codebook_dim, emb_dim):
    sphere = Hypersphere(dim=emb_dim - 1)
    points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=codebook_dim))
    return points_in_manifold


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p = 2)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)




def unique_sampling_fn(distances, nunique=-1):
    # distance: Bxntokensxncbtokens

    B, S, N = distances.shape
    if not (isinstance(nunique, list) or isinstance(nunique, np.ndarray)):
        if (nunique == -1):
            nunique = min(S, N)
        nunique = [nunique]*B

    nunique = np.minimum(nunique, N)
    batch_sampled_vectors = []
    for b in range(B):
        distance_vector = distances[b, ...]
        sorted_idx = torch.argsort(distance_vector, dim=-1, descending=False)
        # Create a bin over codebook direction of distance vectors
        # with nunique bins and sample based on that...
        #
        #
        sampled_vectors = []
        sampled_distances = []
        for i in range(S):

            if i < nunique[b]:
                for k in range(N):
                    current_idx = sorted_idx[i, k]
                    if not (current_idx in sampled_vectors):
                        sampled_vectors.append(current_idx.unsqueeze(0))
                        sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
                        break
            else:
                current_idx = sorted_idx[i, 0]
                sampled_vectors.append(current_idx.unsqueeze(0))
                sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))


        sampled_vectors = torch.cat(sampled_vectors, 0)
        sampled_distances = torch.cat(sampled_distances, 0)
        sampled_vectors = sampled_vectors[torch.argsort(sampled_distances, descending=False)]
        batch_sampled_vectors.append(sampled_vectors.unsqueeze(0))

    batch_sampled_vectors = torch.cat(batch_sampled_vectors, 0)
    return batch_sampled_vectors.view(-1)


def get_euclidian_distance(u, v):
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = torch.sum(u**2, dim=1, keepdim=True) + \
        torch.sum(v**2, dim=1) - 2 * torch.matmul(u, v.t())
    return d



def sorting_idxs(quantized, cidxs):
    B, N, _ = quantized.shape

    batch_sampled_vectors = []
    for b in range(B):
        sampled_idxs = cidxs[b, ...]

        st_pointer = -1
        end_pointer = 0
        unique = torch.zeros_like(sampled_idxs)
        for unique_idx in torch.sort(torch.unique(sampled_idxs)):
            idxs = torch.argwhere(sampled_idxs == unique_idx)

            st_pointer += 1
            end_pointer -= len(idxs[1:])

            unique[st_pointer] = idxs[0]
            unique_idx[end_pointer : end_pointer + len(idxs[1:])] = idxs[1:]

            pass


    return quantized, cidxs


def get_cosine_distance(u, v):
    # distance on sphere
    d = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
    ed1 = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
    ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
    ed3 = torch.einsum('bd,dn->bn', ed1, rearrange(ed2, 'n d  -> d n'))
    geod = torch.clamp(d/(ed3), min=-0.99999, max=0.99999)
    return torch.acos(torch.abs(geod))/(2.0*np.pi)


def get_cb_variance(cb):
    # cb = cb / (1e-5 + torch.norm(cb, dim=1, keepdim=True))
    cd = get_cosine_distance(cb, cb)
    return 1 - torch.mean(torch.var(cd, 1))







# Implementations adopted from https://github.com/lukemelas/deep-spectral-segmentation
def get_diagonal(W: scipy.sparse.csr_matrix, threshold: float = 1e-12):
    # See normalize_rows in pymatting.util.util
    D = row_sum(W)
    D[D < threshold] = 1.0  # Prevent division by zero.
    D = scipy.sparse.diags(D)
    return D


# Implementations adopted from https://github.com/lukemelas/deep-spectral-segmentation
def knn_affinity(image, n_neighbors=[2, 1], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""

    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    # This is our affinity matrix
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    return W


# Implementations adopted from https://github.com/lukemelas/deep-spectral-segmentation
@torch.no_grad()
def compute_eigen(
    feats,
    image = None,
    K: int = 4,
    which_matrix: str = 'affinity_torch',
    normalize: bool = True,
    binarize: bool = True,
    lapnorm: bool = True,
    threshold_at_zero: bool = False,
    image_color_lambda: float = 10,
):

    if normalize:
        feats = F.normalize(feats, p=2, dim=-1)


    W = feats @ feats.T
    if threshold_at_zero:
        W = (W * (W > 0))


    # if binarize:
    #     # apply softmax on token dimension
    #     W = torch.softmax(W, dim = 1)

        # W = torch.sigmoid(W)
        # W[W >= 0.5] = 1
        # W[W < 0.5] = 0


    # Eigenvectors of affinity matrix
    if which_matrix == 'affinity_torch':
        eigenvalues, eigenvectors = torch.eig(W, eigenvectors=True)
        eigenvalues = eigenvalues[-K:]
        eigenvectors = eigenvectors[:, -K:].T

    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity_svd':
        USV = torch.linalg.svd(W, full_matrices=False)
        eigenvectors = USV[0][:, -K:].T #.to('cpu', non_blocking=True)
        eigenvalues = USV[1][-K:] #.to('cpu', non_blocking=True)

    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity':
        W = W.cpu().numpy()
        eigenvalues, eigenvectors = eigsh(W, which='LM', k=K)

    # Eigenvectors of matting laplacian matrix
    elif which_matrix in ['matting_laplacian', 'laplacian']:

        ### Feature affinities
        W_feat = W.cpu().numpy()

        # Combine
        if image_color_lambda > 0:
            if image is None:
                raise ValueError('Image argument is required for laplacian based eigen decomposition')
            # Load image
            H = int(feats.shape[0]**0.5)
            image_lr = F.interpolate(image.unsqueeze(0),
                size=(H, H), mode='bilinear', align_corners=False
            ).squeeze(0).cpu().numpy().transpose(1,2,0)


            # Color affinities (of type scipy.sparse.csr_matrix)
            W_lr = knn_affinity(image_lr)

            # Convert to dense numpy array
            W_color = np.array(W_lr.todense().astype(np.float32))
            W_color *=0.25
        else:

            # No color affinity
            W_color = 0


        W_comb = W_feat + W_color * image_color_lambda  # combination
        D_comb = np.array(get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

        # Extract eigenvectors
        if lapnorm:
            try:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
            except:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
        else:
            try:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM')
            except:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM')
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    # Sign ambiguity
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]


    # normalization of eigen vectors
    # eigenvectors = eigenvectors/torch.norm(eigenvectors, dim=-1, keepdim=True)

    # Save dict
    # _, indices = torch.sort(eigenvalues, 0)
    # if len(indices.shape) > 1:
    #     indices = indices[:, 0]

    # eigenvalues = eigenvalues[indices]
    # eigenvectors = eigenvectors[indices, :]
    return eigenvectors, eigenvalues



@torch.no_grad()
def plot_vis(img, eigenvectors, K=10):
    n = min(eigenvectors.shape[0], K) + 1
    img = img.cpu().numpy().transpose(1,2,0)

    plt.figure(figsize=(3*n, 3))
    plt.subplot(1, n, 1)
    plt.imshow(img)

    for i in range(1, n):
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.imshow(eigenvectors[i-1].squeeze().cpu().numpy(), cmap='coolwarm', vmax=1.0, vmin=0.0, alpha=0.7)

    plt.show()


def process_eigen(eigenvectors, img=None, visual= False, binarize=0.0):
    eigen_reshape = int(eigenvectors.shape[-1]**0.5)

    print (eigen_reshape, eigenvectors.shape)
    # transpose as col correspond to eigen vectors
    eigenvectors = eigenvectors.T.view(-1, 1, eigen_reshape, eigen_reshape)
    # eigenvectors = (eigenvectors - torch.min(eigenvectors))/ (torch.max(eigenvectors) - torch.min(eigenvectors))

    if visual:
        if img is None:
            raise ValueError()
        _, H, W = img.shape
        eigenvectors = F.interpolate(
                    eigenvectors,
                    size=(H, W), mode='bilinear', align_corners=True
                )

    return eigenvectors



def visual_concepts(eigenvectors, data, binarize=0.0):
    B, c, H,W = data.shape
    b, k, c, h, w = eigenvectors.shape
    visual = []
    for i in range(int(k)):
        conecpts = eigenvectors[:, i, ...]
        resized =  F.interpolate(
                        conecpts,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=True
                    )
        visual.append(resized.unsqueeze(1))
    visual = torch.cat(visual, 1)
    return visual


class QKCodebook(nn.Module):
    def __init__(self, dim, codebook_size):
        super().__init__()
        self.dim = dim

        self.register_buffer('mu_embeddings', hsphere_init(codebook_size, dim))
        self.register_buffer('sigma_embeddings', uniform_init(codebook_size, dim))

        self.fc1_w = nn.Parameter(uniform_init(codebook_size, dim, dim))
        self.fc1_b = nn.Parameter(uniform_init(codebook_size, dim))

        self.fc2_w = nn.Parameter(uniform_init(codebook_size, dim, dim))
        self.fc2_b = nn.Parameter(uniform_init(codebook_size, dim))


    def sample_slots(self, encodings, shape, MCsamples=1):
        # encodings: encoded index MB*Ntokens x 1
        # shape: MB x Ntokens x dim

        shape = (shape[0], shape[1], self.dim)
        slot_mu = torch.matmul(encodings, self.mu_embeddings)
        slot_sigma = torch.matmul(encodings, self.sigma_embeddings)
        slot_sigma = torch.exp(0.5*slot_sigma)
        slots = torch.cat([torch.normal(slot_mu, slot_sigma).view(shape).unsqueeze(1) for _ in range(MCsamples)], 1)


        # weights for transformations =======================
        encodings = encodings.view(shape[0], shape[1], -1)
        fc1w = torch.einsum('bnd,dgk->bngk', encodings, self.fc1_w)
        fc1b = torch.einsum('bnd,dg->bng',encodings, self.fc1_b)

        fc2w = torch.einsum('bnd,dgk->bngk',encodings, self.fc2_w)
        fc2b = torch.einsum('bnd,dg->bng',encodings, self.fc2_b)


        # apply transformation ===============
        slots = F.relu(torch.einsum('bmnd,bndw->bmnw', slots, fc1w) + fc1b.unsqueeze(1))
        slots = F.relu(torch.einsum('bmnd,bndw->bmnw', slots, fc2w) + fc2b.unsqueeze(1))


        # slot_mu = slot_mu.view(shape)
        # slot_sigma = slot_sigma.view(shape)

        # # MC expectation estimation
        # sampling_shape = (slot_sigma.shape[0], MCsamples, slot_sigma.shape[1], slot_sigma.shape[2])
        # slots = slot_mu.unsqueeze(1) + slot_sigma.unsqueeze(1) * torch.randn(sampling_shape,
        #                                                     device = slot_sigma.device,
        #                                                     dtype = slot_sigma.dtype)

        # MC samples along batch axis.....

        slots = rearrange(slots, 'b m n d -> (b m) n d')
        return slots #, slot_mu, slot_sigma


    def compute_qkloss(self, embed_ind, inputs):
        # include all QKcodebook regularizations
        input_shape = inputs.shape

        unique_code_ids = torch.unique(embed_ind)

        qkloss = 0
        # qkloss += get_cb_variance(mu_codebook)

        # kl loss between sampled marginal distributions
        qkloss += 0 # torch.mean(-0.5 * (1 + logvar - mean ** 2 - logvar.exp()))

        return qkloss




class BaseVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings,
                        embedding_dim,
                        nhidden,
                        codebook_dim = 8,
                        commitment_cost=0.25,
                        usage_threshold=1.0e-9,
                        qk=False,
                        cosine=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.codebook_dim = codebook_dim
        self.nhidden = nhidden
        self._cosine = cosine
        self.qk=qk


        requires_projection = codebook_dim != embedding_dim
        self.project_in = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, codebook_dim)) if requires_projection else nn.Identity()
        self.project_out = nn.Sequential(nn.Linear(codebook_dim, embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, embedding_dim)) if requires_projection else nn.Identity()

        self.norm_in  = nn.LayerNorm(codebook_dim)
        self.norm_out  = nn.LayerNorm(embedding_dim)



        self._embedding = nn.Embedding(self._num_embeddings, codebook_dim)
        self._get_distance = get_euclidian_distance
        self.loss_fn = F.mse_loss



        if self._cosine:
            sphere = Hypersphere(dim=codebook_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            self._embedding.weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance
            self.loss_fn = lambda x1,x2: 2.0*(1 - F.cosine_similarity(x1, x2).mean())


        self.data_mean = 0
        self.data_std = 0

        self.register_buffer('_usage', torch.ones(self._num_embeddings), persistent=False)


        self.kld_scale = kld_scale
        # ======================
        # QK codebook init
        if self.qk:
            self.qkclass = QKCodebook(self.nhidden, self._num_embeddings)

        # ======================
        # Gumble parameters
        self.gumble = gumble
        if self.gumble:
            self.temperature = temperature
            self.gumble_proj = nn.Sequential(nn.Linear(codebook_dim, num_embeddings))


    def update_usage(self, min_enc):
        self._usage[min_enc] = self._usage[min_enc] + 1  # if code is used add 1 to usage
        self._usage /= 2 # decay all codes usage

    def reset_usage(self):
        self._usage.zero_() #  reset usage between epochs

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self._usage < self._usage_threshold).squeeze(1)
        useful_codes = torch.nonzero(self._usage > self._usage_threshold).squeeze(1)
        N = self.data_std.shape[0]

        if len(dead_codes) > 0:
            eps = torch.randn((len(dead_codes), self._embedding_dim)).to(self._embedding.weight.device)
            rand_codes = eps*self.data_std.unsqueeze(0).repeat(len(dead_codes), 1) +\
                                self.data_mean.unsqueeze(0).repeat(len(dead_codes), 1)

            with torch.no_grad():
                self._embedding.weight[dead_codes] = rand_codes

            self._embedding.weight.requires_grad = True


    def get_encodings(self, encodings):
        quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)
        return quantized

    def vq_sample(self, features,
                        hard = False,
                        idxs=None,
                        MCsamples = 1,
                        final=False):
        input_shape = features.shape

        def _min_encoding_(distances):
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            sampled_dist, encoding_indices = torch.min(distances, dim=1)
            encoding_indices = encoding_indices.view(input_shape[0], -1)
            sampled_dist = sampled_dist.view(input_shape[0], -1)

            # import pdb;pdb.set_trace()
            encoding_indices = encoding_indices.view(-1)
            # encoding_indices = encoding_indices[torch.argsort(sampled_dist, dim=1, descending=False).view(-1)]
            encoding_indices = encoding_indices.unsqueeze(1)
            return encoding_indices

        # Flatten features
        features = features.view(-1, self.codebook_dim)


        # Update prior with previosuly wrt principle components
        # This will ensure that codebook indicies that are not in idxs wont be sampled
        # hacky way need to find a better way of implementing this
        key_codebook = self._embedding.weight.clone()
        if not (idxs is None):
            idxs = torch.unique(idxs).reshape(-1, 1)
            for i in range(self._num_embeddings):
                if not (i in idxs):
                    key_codebook[i, :] = 2*torch.max(features)


        # Quantize and unflatten
        distances = self._get_distance(features, key_codebook)
        encoding_indices = _min_encoding_(distances)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight)

        # =========
        slots = None
        # slot sampling
        if self.qk:
            slots = self.qkclass.sample_slots(encodings,
                                                input_shape,
                                                MCsamples=MCsamples)


        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)
        quantized = quantized.view(input_shape[0], input_shape[1], -1)
        return quantized, encoding_indices, encodings, slots, None


    def gumble_sample(self, features,
                        hard=False,
                        idxs = None,
                        MCsamples = 1,
                        final=False):

        input_shape = features.shape

        # force hard = True when we are in eval mode, as we must quantize
        logits = self.gumble_proj(features)

        # Update prior with previosuly wrt principle components
        if not (idxs is None):
            mask_idxs = torch.zeros_like(logits)

            for b in range(input_shape[0]):
                mask_idxs[b, :, idxs[b]] = 1
            logits = mask_idxs*logits


        soft_one_hot = F.gumbel_softmax(logits,
                                        tau=self.temperature,
                                        dim=-1, hard=hard)

        quantized = torch.einsum('b t k, k d -> b t d',
                                        soft_one_hot,
                                        self._embedding.weight)



        encoding_indices = soft_one_hot.argmax(dim=-1).view(-1, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)


        # ==========================
        slots = None
        # slot sampling
        if self.qk:
            slots = self.qkclass.sample_slots(encodings,
                                                    input_shape,
                                                    MCsamples=MCsamples)

        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)
        return quantized, encoding_indices, encodings, slots, logits



    def sample(self, features,
                        hard=False,
                        idxs = None,
                        MCsamples = 1,
                        from_train = False):
        if not from_train:
            features = self.project_in(features)
            # layer norm on features
            features = self.norm_in(features)


        if self.gumble:
            return self.gumble_sample(features,
                                        MCsamples=MCsamples,
                                        hard=hard,
                                        idxs=idxs)
        else:
            return self.vq_sample(features,
                                    MCsamples=MCsamples,
                                    hard=hard,
                                    idxs=idxs)


    def compute_baseloss(self, quantized, inputs, logits=None, loss_type=0):
        if self.gumble:
            # + kl divergence to the prior loss
            if (logits is None):
                loss = 0
            else:
                # print (logits.min(), logits.max(), logits.mean(), '===========')
                qy = F.softmax(logits, dim=-1)
                loss = self.kld_scale * torch.sum(qy * torch.log(qy * self._num_embeddings + 1e-10), dim=-1).mean()

        else:
             # Loss
            e_latent_loss = self.loss_fn(quantized.detach(), inputs)
            q_latent_loss = self.loss_fn(quantized, inputs.detach())


            if loss_type == 0:
                loss = q_latent_loss + self._commitment_cost * e_latent_loss
            elif loss_type == 1:
                loss = q_latent_loss
            else:
                loss = self._commitment_cost * e_latent_loss

        return loss


    def forward(self, *args):
        return NotImplementedError()



class VectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embeddings,
                        embedding_dim,
                        codebook_dim=8,
                        nhidden=128,
                        decay=0.8,
                        epsilon=1e-5,
                        commitment_cost=0.0,
                        usage_threshold=1.0e-9,
                        qk=False,
                        cosine=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0):

        super(VectorQuantizer, self).__init__(num_embeddings = num_embeddings,
                                                    embedding_dim = embedding_dim,
                                                    codebook_dim = codebook_dim,
                                                    nhidden = nhidden,
                                                    commitment_cost = commitment_cost,
                                                    usage_threshold = usage_threshold,
                                                    qk = qk,
                                                    cosine = cosine,
                                                    gumble = gumble,
                                                    temperature = temperature,
                                                    kld_scale = kld_scale)

        self._embedding_dim = embedding_dim
        self.codebook_dim = codebook_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self._cosine = cosine
        self.qk = qk



        # ======================
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.codebook_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs,
                        update=False,
                        loss_type=0,
                        idxs=None,
                        MCsamples = 1,
                        reset_usage=False):
        input_shape = inputs.shape

        features = self.project_in(inputs)
        # layer norm on features
        features = self.norm_in(features)

        # Flatten input
        flat_input = features.view(-1, self.codebook_dim)


        # update data stats
        if self.training:
            self.data_mean = 0.9*self.data_mean + 0.1*features.clone().detach().mean(1).mean(0)
            self.data_std = 0.9*self.data_std + 0.1*features.clone().detach().std(1).mean(0)



        quantized, encoding_indices, encodings, slots, logits = self.sample(features,
                                                                            idxs=idxs,
                                                                            hard = True,
                                                                            MCsamples = MCsamples,
                                                                            from_train=True)



        # ===================================
        # Tricks to prevent codebook collapse---
        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()


        # Reset unused cb vectors...
        if update: self.update_usage(encoding_indices)


        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))


        # ============================
        loss = self.compute_baseloss(quantized, inputs, logits, loss_type)
        unique_code_ids = torch.unique(encoding_indices)




        # Regularization terms ==========================
        # loss += get_cb_variance(self._embedding.weight[unique_code_ids])

        if not (loss_type == 2):
            if self.qk:
                qkloss = self.qkclass.compute_qkloss(encoding_indices,
                                                            inputs)

                loss += qkloss


        # print (f'feature: {inputs.max()}, qfeatures: {quantized.max(), quantized.min(), quantized.mean()}, quant loss: {loss}, QKloss: {qkloss}')

        # Straight Through Estimator
        if not self.gumble: quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(input_shape[0], -1)

        return quantized, encoding_indices, loss, perplexity, slots




def reorder_slots(slots, slots_mu, cidxs, scales = None, ns=10):
    # eigenvalues in decreasing order
    # cidxs are ordered wrt eigenvalues

    B, N = cidxs.shape
    if ns > N:
        orig_slots = slots.clone()
        orig_slots_mu = slots_mu.clone()
        orig_idxs = cidxs.clone()

        counter = 1
        while cidxs.shape[1] < ns:
            nunique_objects = -1
            if not (scales is None):
                nunique_objects = int((1.0*(scales > counter).sum(1)).max().item())

            start_idx = 0
            if nunique_objects > 1:
                start_idx = 1

            slots = torch.cat([slots, orig_slots[:, start_idx:nunique_objects, :]], 1)
            slots_mu = torch.cat([slots_mu, orig_slots_mu[:, start_idx:nunique_objects, :]], 1)
            cidxs = torch.cat([cidxs, orig_idxs[:, start_idx:nunique_objects]], 1)

            counter += 1


    slots, slots_mu, cidxs = slots[:, :ns, :], slots_mu[:, :ns, :], cidxs[:, :ns]

    return slots, slots_mu, cidxs



class SlotAttention(nn.Module):
    def __init__(self, num_slots,
                        dim,
                        iters = 3,
                        eps = 1e-8,
                        hidden_dim = 128,
                        max_slots=64,
                        nunique_slots=8,
                        beta=0.0,
                        encoder_intial_res=(8, 8),
                        decoder_intial_res=(8, 8),
                        cosine=False,
                        cb_decay=0.8,
                        cb_querykey=False,
                        eigen_quantizer=False,
                        restart_cbstats=False,
                        implicit=True,
                        gumble=False,
                        temperature=2.0,
                        kld_scale=1.0
                        ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.scale = dim ** -0.5
        self.implicit = implicit
        self.cb_querykey = cb_querykey
        self.eigen_quantizer = eigen_quantizer
        self.restart_cbstats = restart_cbstats
        self.resolution = encoder_intial_res
        ntokens = np.prod(encoder_intial_res)

        self.max_slots = max_slots
        self.nunique_slots = nunique_slots
        self.min_number_elements = 2
        self.beta = beta
        legacy = True

        assert self.num_slots <= np.prod(encoder_intial_res), f'reduce number of slots, max possible {np.prod(encoder_intial_res)}'

        # ===========================================
        # encoder postional embedding with linear transformation
        self.encoder_position = SoftPositionEmbed(dim, encoder_intial_res)
        self.encoder_norm = nn.LayerNorm([ntokens, dim])
        self.encoder_feature_mlp = nn.Sequential(nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True))


        self.slot_transformation = nn.Sequential(nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(dim, dim),
                                                nn.ReLU(inplace=True))



        # # decoder positional embeddings
        self.decoder_position    = SoftPositionEmbed(dim, decoder_intial_res)
        self.decoder_initial_res = decoder_intial_res

        # ===========================================

        self.to_v = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.to_q = nn.Linear(dim, dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_input  = nn.LayerNorm(dim)

        # ===================================
        self.slot_quantizer = VectorQuantizer(num_embeddings = max_slots,
                                                        embedding_dim = self.dim, # ntokens,
                                                        codebook_dim = 512,
                                                        nhidden = self.dim,
                                                        commitment_cost = self.beta,
                                                        decay = cb_decay,
                                                        qk=self.cb_querykey,
                                                        cosine= cosine,
                                                        gumble=gumble,
                                                        temperature=temperature
                                                        )

        print ('VQVAE model', cb_decay, cosine, self.dim)


    def encoder_transformation(self, features, position=True):
        #features: B x C x Wx H
        features = features.permute(0, 2, 3, 1)
        if position:
            features = self.encoder_position(features)
            features = torch.flatten(features, 1, 2)
            features = self.encoder_norm(features)
            features = self.encoder_feature_mlp(features)
        else:
            features = torch.flatten(features, 1, 2)
        return features


    def decoder_transformation(self, slots):
        # features: B x nslots x dim
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        features = slots.repeat((1, self.decoder_initial_res[0], self.decoder_initial_res[1], 1))
        features = self.decoder_position(features)
        return features.permute(0, 3, 1, 2)


    @torch.no_grad()
    def passthrough_eigen_basis(self, x):
        # x: token embeddings B x ntokens x token_embeddings

        x = F.normalize(x, p=2, dim=-1)
        coveriance_matrix = torch.einsum('bkf, bgf -> bkg', x, x)

        # eigen vectors are arranged in ascending order of their eigen values
        # eigen_values, eigen_vectors = torch.symeig(coveriance_matrix, eigenvectors=True)
        eigen_values, eigen_vectors = torch.linalg.eigh(coveriance_matrix)

        eigen_vectors = eigen_vectors.permute(0, 2, 1)

        eigen_vectors = torch.flip(eigen_vectors, [1])
        eigen_values = torch.flip(eigen_values, [1])

        max_indices = torch.argmax(torch.abs(eigen_vectors), dim=1)  # Shape: (batch_size, n)

        # Create a batch index tensor
        batch_indices = torch.arange(coveriance_matrix.shape[0], device=coveriance_matrix.device).view(-1, 1)  # Shape: (batch_size, 1)

        # Gather the signs of the elements at the max indices
        signs = torch.sign(eigen_vectors[batch_indices, max_indices, torch.arange(eigen_vectors.shape[2])])  # Shape: (batch_size, n)

        # Reshape signs for broadcasting and apply them to eigen_vectors
        eigen_vectors = eigen_vectors * signs.unsqueeze(1)  # Shape: (batch_size, n, n)


        return eigen_vectors, eigen_values



    @torch.no_grad()
    def extract_eigen_basis(self, features, batch=None):
        batched_concepts = []
        batched_scale = []

        shape = features.shape


        for i, feature in enumerate(features):
            eigen_vectors, eigen_values = compute_eigen(feature,
                                        image = None if batch is None else batch[i],
                                        K = shape[1],# self.nunique_slots+1,
                                        which_matrix = 'matting_laplacian',
                                        normalize  = True,
                                        binarize = self.cov_binarize,
                                        lapnorm = True,
                                        threshold_at_zero = True,
                                        image_color_lambda = 0 if batch is None else 1.0)

            batched_concepts.append(eigen_vectors.unsqueeze(0))
            batched_scale.append(eigen_values.unsqueeze(0))

        batched_scale = torch.cat(batched_scale, 0).to(features.device)
        batched_concepts = torch.cat(batched_concepts, 0).to(features.device)

        batched_concepts = batched_concepts.softmax(dim = -1)

        batched_concepts = torch.flip(batched_concepts, [1])
        batched_scale = torch.flip(batched_scale, [1])
        return batched_concepts, batched_scale


    def masked_projection(self, features, z):
        # features: B x nanchors x f
        # z: basis B x K x nanchors

        # b, n, f = features.shape
        # k = z.shape[1]
        # features = features.unsqueeze(1).repeat(1, k, 1, 1) # B x k x n x f
        # z = z.unsqueeze(-1)
        # z = z.repeat(1, 1, 1, f)# B x k x n x f

        # projection = features*z
        # return torch.sum(projection, 2)

        return torch.bmm(z, features)


    def feature_abstraction(self, inputs):
        # Compute Principle directions and scale
        eigen_basis, eigen_values = self.passthrough_eigen_basis(inputs.clone().detach())
        # eigen_basis, eigen_values = self.extract_eigen_basis(inputs, batch=images)

        eigen_values = torch.round(eigen_values)
        nunique_objects = max(3, int((1.0*(eigen_values > 0).sum(1)).max().item()))

        # Principle components
        objects = self.masked_projection(inputs, eigen_basis)
        objects = objects[:, :nunique_objects, :]
        return objects, eigen_values[:, :nunique_objects]


    def sample_quantized_slots(self, n_s, k, MCsamples = 1, epoch = 0, batch = 0, images=None):

        # get principle components
        if self.eigen_quantizer: objects, scales = self.feature_abstraction(k)
        else: objects = k; scales = torch.ones_like(k)

        # Quantizing components----
        qobjects, cbidxs, qloss, perplexity, slots  = self.slot_quantizer(objects.detach(),
                                                                    loss_type = 1,
                                                                    MCsamples = MCsamples,
                                                                    update = self.restart_cbstats,
                                                                    reset_usage = (batch == 0))

        if self.cb_querykey:
            slots, qobjects, cbidxs = reorder_slots(slots, qobjects, cbidxs, scales, n_s)
            slots = slots.reshape(k.shape[0], MCsamples, -1, slots.shape[-1])
        else:
            qobjects = objects[:, :n_s, :]
            slots = qobjects.clone().unsqueeze(1); cbidxs = cbidxs[:, :n_s]

        return slots, qobjects, qloss, perplexity, cbidxs


    def sample_baseline_slots(self, inputs, n_s, b):
        qloss = torch.Tensor([0]).to(inputs.device)
        cbidxs = torch.Tensor([[0]*n_s]*b).to(inputs.device)
        perplexity = torch.Tensor([[0]]).to(inputs.device)

        slot_mu = self.slots_mu.expand(b, n_s, -1)
        slot_sigma = self.slots_sigma.expand(b, n_s, -1)
        slot_sigma = torch.exp(0.5*slot_sigma)

        slots = slot_mu + slot_sigma * torch.randn(slot_sigma.shape,
                                                device = slot_sigma.device,
                                                dtype = slot_sigma.dtype)

        # slots = torch.normal(slot_mu, slot_sigma)

        # add MC axis
        slots = slots.unsqueeze(1)
        return slots, slot_mu.clone(), qloss, perplexity, cbidxs


    def step(self, slots_prev, k , v, MCsamples, ns, b):
        slots = self.norm_slots(slots_prev)
        q = self.to_q(slots)

        dots = torch.einsum('bmid,bjd->bmij', q, k) * self.scale
        attn = dots.softmax(dim=2) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bmij->bmid', v, attn)

        slots = self.gru(
            rearrange(updates, 'b m n d -> (b m n) d'),
            rearrange(slots_prev, 'b m n d -> (b m n) d')
        )

        slots = slots.reshape(b, MCsamples, ns, self.dim)
        slots = slots + self.slot_transformation(self.norm_pre_ff(slots))
        return slots


    def forward(self, inputs,
                    num_slots = None,
                    MCsamples = 1,
                    epoch=0, batch= 0,
                    train = True,
                    images=None):

        b, d, w, h = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # Compute Projections ========================
        inputs_features = self.encoder_transformation(inputs, position = True)
        inputs_features = self.norm_input(inputs_features)


        # Sample Slots ========================
        inputs_features_np = self.encoder_transformation(inputs, position = False)

        # quantize and sample slots wrt principle components
        slots, objects, qloss, perplexity, cbidxs = self.sample_quantized_slots(n_s = n_s,
                                                                        MCsamples = MCsamples,
                                                                        k = inputs_features_np,
                                                                        epoch = epoch,
                                                                        batch = batch,
                                                                        images = images)




        # Key-Value projection vectors ====================
        k = self.to_k(inputs_features)
        v = self.to_v(inputs_features)

        # Slot attention =========================
        for _ in range(self.iters):
            slots = self.step(slots, k, v, MCsamples, n_s, b)

        if self.implicit: slots = self.step(slots.detach(), k, v, MCsamples, n_s, b)


        return slots, cbidxs, qloss, perplexity, self.decoder_transformation(slots)



    @torch.no_grad()
    def given_idxs(self, slot_idxs,
                        n_s = 7,
                        MCsamples = 1,
                        images=None):

        b = slot_idxs.shape[0]
        d = self.dim
        w,h = self.resolution

        # sample slots ==============================
        qloss = torch.Tensor([0]).to(slot_idxs.device)
        perplexity = torch.Tensor([[0]]).to(slot_idxs.device)

        shape = slot_idxs.shape
        encodings = torch.zeros(np.prod(shape), self.max_slots, device=slot_idxs.device)
        encodings.scatter_(1, slot_idxs.reshape(-1, 1), 1)


        # Input features ===========================
        inputs = self.slot_quantizer.get_encodings(encodings)
        inputs += torch.mean(torch.norm(inputs, 1))*0.05*torch.randn_like(inputs)

        inputs = inputs.view(b, w, h, d).permute(0,3,1,2)
        inputs = self.encoder_transformation(inputs, position=True)
        inputs = self.norm_input(inputs)


        # Slot slection =======================
        slots = self.slot_quantizer.qkclass.sample_slots(encodings, (b, w*h, self.dim))
        slots = slots.view(b, MCsamples, -1, d)
        slots = slots[:, :, :n_s, :]


        # # Key-Value projection vectors ====================
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        # # Slot attention =========================
        for _ in range(self.iters):
            slots = self.step(slots, k, v, MCsamples, n_s, b)

        if self.implicit: slots = self.step(slots.detach(), k, v, MCsamples, n_s, b)

        return slots, slot_idxs, qloss, perplexity, self.decoder_transformation(slots)




def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)



"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T])
