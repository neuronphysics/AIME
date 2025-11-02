import torch
import torch.nn as nn
import os, sys
import numpy as np
import einops
from sklearn.preprocessing import RobustScaler
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
    """
    Normalizes variable-length sequences with zero padding preserved.
    Data shape: (batch_size, feature_dim, seq_len)
    """
    def __init__(self, inputs, clip_range=(-5, 5)):
        super(Normalizer1D, self).__init__()
        self.clip_range = clip_range
        self.input_dim = inputs.shape[1]
        self.max_seq = inputs.shape[2]
        
        # Store device for consistent tensor operations
        self.device = inputs.device
        
        # Create padding masks - identify which values are originally zero
        self.register_buffer('padding_mask', (inputs == 0))
        
        # Build and store normalizers
        self._norm = self.build_normalizers(inputs)
        self.to(self.device)

    def build_normalizers(self, x):
        """Build a normalizer for each feature dimension across all timesteps"""
        normalizers = []
        scale_factors = []
        centers = []
        
        for i in range(self.input_dim):
            # Collect all non-zero values for this feature across all timesteps
            valid_values = []
            for seq in range(self.max_seq):
                x_data = x[:, i, seq]
                non_zero_mask = (x_data != 0)
                if non_zero_mask.sum() > 0:
                    valid_values.append(x_data[non_zero_mask].cpu().numpy())
            
            # Fit scaler to valid values
            if len(valid_values) > 0:
                valid_values = np.concatenate(valid_values).reshape(-1, 1)
                scaler = RobustScaler()
                scaler.fit(valid_values)
                scale_factors.append(torch.tensor(scaler.scale_[0], device=self.device))
                centers.append(torch.tensor(scaler.center_[0], device=self.device))
            else:
                # If no valid values, use identity transformation
                scaler = RobustScaler()
                scaler.fit(np.array([[0], [1]]))
                scale_factors.append(torch.tensor(1.0, device=self.device))
                centers.append(torch.tensor(0.0, device=self.device))
            
            normalizers.append(scaler)
        
        # Register buffers for scale factors and centers
        self.register_buffer('scale_factors', torch.stack(scale_factors))
        self.register_buffer('centers', torch.stack(centers))
        
        return normalizers

    def normalize(self, x):
        """Normalize input tensor while preserving zero padding"""
        # Create result tensor
        result = torch.zeros_like(x, device=self.device)
        non_zero_mask = (x != 0)
        
        # Process each feature
        for i in range(x.shape[1]):
            feature_mask = non_zero_mask[:, i, :]
            
            if not feature_mask.any():
                continue  # Skip features with all zeros
            
            # Process each timestep
            for t in range(x.shape[2]):
                time_mask = feature_mask[:, t]
                
                if not time_mask.any():
                    continue  # Skip timesteps with all zeros
                
                # Get values and convert to NumPy
                values = x[time_mask, i, t].cpu().numpy().reshape(-1, 1)
                
                # Transform values
                transformed = self._norm[i].transform(values).flatten()
                
                # Clip to ensure stable range
                transformed = np.clip(transformed, self.clip_range[0], self.clip_range[1])
                
                # Convert back to tensor and assign
                result[time_mask, i, t] = torch.tensor(
                    transformed, 
                    dtype=x.dtype, 
                    device=self.device
                )
        
        return result

    def unnormalize(self, x):
        """Unnormalize tensor while preserving zero padding"""
        # Create result tensor
        result = torch.zeros_like(x, device=self.device)
        non_zero_mask = (x != 0)
        
        # Process each feature
        for i in range(x.shape[1]):
            feature_mask = non_zero_mask[:, i, :]
            
            if not feature_mask.any():
                continue  # Skip features with all zeros
            
            # Process each timestep
            for t in range(x.shape[2]):
                time_mask = feature_mask[:, t]
                
                if not time_mask.any():
                    continue  # Skip timesteps with all zeros
                
                # Get values and convert to NumPy
                values = x[time_mask, i, t].cpu().numpy().reshape(-1, 1)
                
                # Inverse transform values
                untransformed = self._norm[i].inverse_transform(values).flatten()
                
                # Convert back to tensor and assign
                result[time_mask, i, t] = torch.tensor(
                    untransformed, 
                    dtype=x.dtype, 
                    device=self.device
                )
        
        return result

    def unnormalize_mean(self, x_mu):
        """Unnormalize mean values - delegates to unnormalize"""
        return self.unnormalize(x_mu)

    def unnormalize_sigma(self, x_sigma):
        """
        Unnormalize standard deviations by scaling only (no center adjustment)
        This is optimized to use vectorized operations
        """
        result = torch.zeros_like(x_sigma, device=self.device)
        non_zero_mask = (x_sigma != 0)
        
        # Process each feature with vectorized operations
        for i in range(x_sigma.shape[1]):
            feature_mask = non_zero_mask[:, i, :]
            
            if feature_mask.any():
                # Scale by the robust scaler's scale factor
                result[:, i, :][feature_mask] = x_sigma[:, i, :][feature_mask] * self.scale_factors[i]
        
        return result

# Update compute_normalizer function with default clip range
def compute_normalizer(feat, out, clip_range=(-5, 5)):
    """Create normalizers for both input features and output targets"""
    # input shape (batch_size, feat_dim, seq_len)
    inputs = feat
    outputs = out

    # Create normalizers with the same clip range
    u_normalizer = Normalizer1D(inputs, clip_range=clip_range)
    y_normalizer = Normalizer1D(outputs, clip_range=clip_range)

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
