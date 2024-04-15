import torch
import torch.nn as nn
import os, sys
import numpy as np
import einops

from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mask_from_sequence_lengths(
        sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


class MaskedNormalizer1D(nn.Module):
    # Data size of (batch_size, input_size, seq_len)
    def __init__(self, input_dim, inputs):
        super(MaskedNormalizer1D, self).__init__()
        self.input_dim = input_dim

        self._mask, self._masked_inputs = self._build_mask(inputs)
        self._norm = self.build_normalizers(self._masked_inputs)

        self.to(device)

    def _build_mask(self, x):
        # input should be (batch, dim)
        if torch.is_tensor(x):
            data_ = x.clone()
            new_masked_data = np.zeros_like(x.cpu().detach().numpy())
        else:
            data_ = torch.from_numpy(x)
            new_masked_data = np.zeros_like(x)
        max_len = x.shape[-1]
        length = torch.LongTensor(
            [torch.max((data_[i, 0, :] != 0).nonzero()).item() + 1 for i in range(data_.shape[0])])
        mask = get_mask_from_sequence_lengths(length, max_len)
        mask = mask.detach().numpy()
        for i in range(x.shape[1]):
            M = np.ma.masked_array(data=x[:, i, :], mask=~mask, fill_value=np.nan)
            new_masked_data[:, i, :] = M.filled()
        return mask, new_masked_data

    def build_normalizers(self, x):
        normalizers = []
        scale_, min_ = [], []
        for i in range(self.input_dim):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(x[:, i, :])
            scale_.append(torch.from_numpy(scaler.scale_))
            min_.append(torch.from_numpy(scaler.min_))
            normalizers.append(scaler)
        self.scale_ = torch.stack(scale_).to(device)
        self.min_ = torch.stack(min_).to(device)
        return normalizers

    def normalize(self, x):
        # (B, D, T)
        d = x.cpu().detach().numpy()
        _, nd = self._build_mask(d)
        n_x = []
        for i in range(x.shape[1]):
            n_x.append(self._norm[i].transform(nd[:, i, :]))
        y = np.stack(n_x, axis=1)
        x = np.where(np.isnan(y), 0, y)
        return torch.from_numpy(x).to(device)

    def unnormalize(self, x):
        # (T, B, D)
        d = x.cpu().detach().numpy()
        m, nd = self._build_mask(d)
        self.unnorm_mask = m
        n_x = []
        for i in range(x.shape[1]):
            n_x.append(self._norm[i].inverse_transform(nd[:, i, :]))

        y = np.stack(n_x, axis=1)
        x = np.where(np.isnan(y), 0, y)
        return torch.from_numpy(x).to(device)

    def unnormalize_mean(self, x_mu):
        d = x_mu.clone()
        normX = d.sub_(self.min_)
        return normX.div_(self.scale_)

    def unnormalize_sigma(self, x_sigma):
        d = x_sigma.clone()
        return d.div_(self.scale_)


class Normalizer1D(nn.Module):
    # Data size of (batch_size, input_size, seq_len)
    def __init__(self, inputs):
        super(Normalizer1D, self).__init__()
        self.min_ = None
        self.scale_ = None
        self.input_dim = inputs.shape[-1]
        self.max_seq = inputs.shape[1]

        self._norm = self.build_normalizers(inputs)

        self.to(device)

    def build_normalizers(self, x):
        normalizers = []
        scale_, min_ = [], []

        for seq in range(self.max_seq):
            norm_seq, scale_seq, min_seq = [], [], []
            for i in range(self.input_dim):
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler = scaler.fit(x[:, seq, i].unsqueeze(-1))
                scale_seq.append(torch.from_numpy(scaler.scale_))
                min_seq.append(torch.from_numpy(scaler.min_))
                norm_seq.append(scaler)
            normalizers.append(norm_seq)
            scale_.append(scale_seq)
            min_.append(min_seq)
        self.scale_ = torch.tensor(scale_).to(device)
        self.min_ = torch.tensor(min_).to(device)
        return normalizers

    def normalize(self, x):
        # (B, D, T)
        batch, dim, seq = x.shape
        d = x.cpu().detach()
        reshape_input = d.transpose(2, 1)

        n_x = []
        for seq_c in range(seq):
            n_x_seq = []
            for i in range(x.shape[1]):
                n_x_seq.append(self._norm[seq_c][i].transform(reshape_input[:, seq_c, i].unsqueeze(-1).numpy()))
            n_x.append(np.stack(n_x_seq, axis=1))
        y = np.stack(n_x, axis=0)
        x = np.where(np.isnan(y), 0, y)
        res = torch.from_numpy(x).to(device).squeeze(-1)
        return res.permute(1, 2, 0)

    def unnormalize(self, x):
        # (T, B, D)
        batch, dim, seq = x.shape
        d = x.cpu().detach()
        reshape_input = d.transpose(2, 1)

        n_x = []
        for seq_c in range(seq):
            n_x_seq = []
            for i in range(x.shape[1]):
                n_x_seq.append(self._norm[seq_c][i].inverse_transform(reshape_input[:, seq_c, i].unsqueeze(-1).numpy()))
            n_x.append(np.stack(n_x_seq, axis=1))
        y = np.stack(n_x, axis=0)
        x = np.where(np.isnan(y), 0, y)
        res = torch.from_numpy(x).to(device).squeeze(-1)
        return res.permute(1, 2, 0)

    def unnormalize_mean(self, x_mu):
        d = x_mu.clone()
        batch_size, latent, cur_seq = d.shape
        tight_min = self.min_[:cur_seq, :]
        tight_scale = self.scale_[:cur_seq, :]

        normX = torch.sub(d, tight_min.transpose(0, 1))
        return torch.div(normX, tight_scale.transpose(0, 1))

    def unnormalize_sigma(self, x_sigma):
        d = x_sigma.clone()
        batch_size, latent, cur_seq = d.shape
        tight_scale = self.scale_[:cur_seq, :]

        return torch.div(d, tight_scale.transpose(0, 1))


# compute the normalizers
def compute_normalizer(feat, out):
    # input shape (batch_size, seq_len, feat_dim)
    inputs = feat
    outputs = out

    # initialization
    u_normalizer = Normalizer1D(inputs)
    y_normalizer = Normalizer1D(outputs)

    return u_normalizer, y_normalizer


if __name__ == "__main__":
    total_size = 16
    max_sequence = 15
    latent_dim = 11
    batch = 4

    random_data = torch.rand((total_size, latent_dim, max_sequence))
    old_n = MaskedNormalizer1D(latent_dim, random_data)
    new_n = Normalizer1D(random_data)

    test_data = random_data[:batch, :, :]
    old_out = old_n.normalize(test_data)

    new_out = new_n.normalize(test_data)

    for t in range(batch):
        for l in range(latent_dim):
            for s in range(max_sequence):
                if abs(old_out[t, l, s] - new_out[t, l, s]) > 1e-6:
                    print("Error: res does not match")

    print("test un-normalize")
    old_res = old_n.unnormalize(old_out)
    new_res = new_n.unnormalize(new_out)

    for t in range(batch):
        for l in range(latent_dim):
            for s in range(max_sequence):
                if abs(old_res[t, l, s] - new_res[t, l, s]) > 1e-6:
                    print("Error: res does not match")

    print("finished")
