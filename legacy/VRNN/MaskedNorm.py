import torch
import torch.nn as nn


class CustomLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(CustomLayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.register_parameter("gamma", self.gamma)
            self.beta = nn.Parameter(torch.zeros(num_features))
            self.register_parameter("beta", self.beta)

    def forward(self, x):

        if x.shape[0] == 1:
            # input (batch, sequence, feature)
            mean = x.mean(1)
            std = x.std(1)
            y = (x - mean) / (std + self.eps)
            if self.affine:
                y = self.gamma * y + self.beta
        else:

            # input (batch*not masked sequence, feature)

            shape = [-1] + [1] * (x.dim() - 1)
            mean = x.mean(0, keepdim=True).view(*shape)
            std = x.std(0, keepdim=True).view(*shape)
            assert not torch.isnan(mean).any()
            assert not torch.isinf(mean).any()
            assert not torch.isnan(std).any()
            assert not torch.isinf(std).any()

            y = (x - mean.squeeze(-1)) / (std.squeeze(-1) + self.eps)
            assert not torch.isnan(y).any()
            if self.affine:
                shape = [1, -1] + [1] * (x.dim() - 2)
                y = self.gamma.view(*shape) * y + self.beta.view(*shape)
                assert not torch.isnan(y).any()
        return y


class MaskedNorm(nn.Module):
    # BatchNorm1d with mask
    def __init__(self, num_features, mask_on=True, norm_name=None, affine=True):
        """y is the input tensor of shape [batch_size,  time_length,n_channels]
            mask is of shape [batch_size, 1, time_length]
        """
        # based on https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
        # based on https://github.com/omarsou/MIL-lymphocytosis/blob/6b897605328adbd0ece1b9a5c93dac1e84a80855/mil_framework/embedding_layer.py
        # So the plan is:
        #  1. Merge the batch and time axes using reshape
        #  2. Create a dummy time axis at the end with size 1.
        #  3. Select the valid time steps using the mask
        #  4. Apply BatchNorm1d to the valid time steps
        #  5. Scatter the resulting values to the corresponding positions
        #  6. Unmerge the batch and time axes
        super().__init__()
        if norm_name == "BN":
            self.norm = nn.BatchNorm1d(num_features)
        else:
            self.norm = CustomLayerNorm(num_features, affine=affine, eps=1e-04)
        self.num_features = num_features
        self.mask_on = mask_on
        self._norm_name = norm_name
        #

    def forward(self, y, mask=None):
        # #input BatchNorm (N,C,L)
        # LayerNorm: batch, sentence_length, embedding_dim
        self.sequence_length = y.shape[1]
        if mask is None and self.mask_on:
            seq_len = [torch.max((y[i, :, 0] != 0).nonzero()).item() + 1 for i in range(y.shape[0])]
            m = torch.zeros([y.shape[0], y.shape[1] + 1], dtype=torch.bool).to(y.device)
            m[(torch.arange(y.shape[0]), seq_len)] = 1
            m = m.cumsum(dim=1)[:, :-1]
            mask = (1 - m)
        if self.mask_on:

            if self._norm_name == "BN":
                reshaped = y.reshape([-1, self.num_features, 1])
                reshaped_mask = mask.reshape([-1, 1, 1]) > 0
                selected = torch.masked_select(reshaped, reshaped_mask).reshape([-1, self.num_features, 1])
                batch_normed = self.norm(selected)
                scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
                return scattered.reshape([-1, self.sequence_length, self.num_features])
            else:
                reshaped = y.reshape([-1, self.num_features, 1])
                reshaped_mask = mask.reshape([-1, 1, 1]) > 0
                selected = torch.masked_select(reshaped, reshaped_mask).reshape([-1, self.num_features, 1])
                assert not torch.isnan(selected).any()
                assert not torch.isinf(selected).any()
                batch_normed = self.norm(selected)
                assert not torch.isnan(batch_normed).any()
                scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
                return scattered.reshape([-1, self.sequence_length, self.num_features])
        else:
            if self._norm_name == "BN":
                reshaped = y.reshape([-1, self.num_features, 1])
                batched_normed = self.norm(reshaped)
                return batched_normed.reshape([-1, self.sequence_length, self.num_features])
            else:
                reshaped = y.reshape([-1, self.num_features, 1])
                batch_normed = self.norm(y)
                return batched_normed.reshape([-1, self.sequence_length, self.num_features])
