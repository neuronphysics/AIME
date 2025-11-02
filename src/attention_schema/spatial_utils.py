"""
Spatial Utility Modules for Attention

Contains utility classes for spatial attention processing, including
convolutional GRU cells for temporal dynamics modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU Cell for spatial temporal modeling.

    Implements GRU dynamics over spatial feature maps, allowing the model
    to track spatial attention patterns over time.

    Args:
        input_size (int): Number of input channels
        hidden_size (int): Number of hidden channels
        kernel_size (int): Size of convolutional kernel
        cuda_flag (bool): Whether to use CUDA

    Input:
        input: [B, input_size, H, W] - Input feature map
        hidden: [B, hidden_size, H, W] - Hidden state (optional)

    Output:
        next_h: [B, hidden_size, H, W] - Updated hidden state

    Example:
        >>> conv_gru = ConvGRUCell(input_size=1, hidden_size=32, kernel_size=5, cuda_flag=True)
        >>> x = torch.randn(2, 1, 21, 21)  # Attention map
        >>> h = conv_gru(x)
        >>> print(h.shape)  # [2, 32, 21, 21]
    """

    def __init__(self, input_size, hidden_size, kernel_size, cuda_flag):
        super(ConvGRUCell, self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, kernel_size, padding=self.kernel_size//2)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, kernel_size, padding=self.kernel_size//2)
        dtype            = torch.FloatTensor
        self.norm_gates = nn.GroupNorm(2, 2 * hidden_size)
        self.norm_candidate = nn.GroupNorm(8, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier initialization"""
        # Xavier initialization for gates
        nn.init.xavier_uniform_(self.ConvGates.weight, gain=0.5)
        nn.init.xavier_uniform_(self.Conv_ct.weight, gain=0.5)

        # Initialize biases to favor forgetting initially (stability)
        nn.init.constant_(self.ConvGates.bias, 0.0)
        nn.init.constant_(self.Conv_ct.bias, 0.0)

    def forward(self, input: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of ConvGRU cell.

        Args:
            input: [B, input_size, H, W] - Input feature map
            hidden: [B, hidden_size, H, W] - Previous hidden state (optional)

        Returns:
            next_h: [B, hidden_size, H, W] - Updated hidden state
        """
        if hidden is None:
           size_h    = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
           # Use the same device as input
           hidden    = torch.zeros(size_h, device=input.device, dtype=input.dtype)
        c1           = self.norm_gates(self.ConvGates(torch.cat((input, hidden), 1)))
        (rt, ut)      = c1.chunk(2, 1)
        reset_gate   = F.sigmoid(rt)
        update_gate  = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1           = self.norm_candidate(self.Conv_ct(torch.cat((input, gated_hidden), 1)))
        ct           = F.tanh(p1)
        next_h       = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h
