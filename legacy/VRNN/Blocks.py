import torch
import torch.nn as nn
import numpy as np
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m,nn.GRU) or isinstance(m,nn.LSTM):
        for ind in range(0, m.num_layers):
            weight = eval('m.weight_ih_l'+str(ind))
            bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('m.weight_hh_l'+str(ind))
            bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
        if m.bias:
            for ind in range(0, m.num_layers):
                weight = eval('m.bias_ih_l'+str(ind))
                weight.data.zero_()
                weight.data[m.hidden_size: 2 *m.hidden_size] = 1
                weight = eval('m.bias_hh_l'+str(ind))
                weight.data.zero_()
                weight.data[m.hidden_size: 2 * m.hidden_size] = 1
    elif isinstance(m, list):
        for layer in m:
            if isinstance(layer, nn.BatchNorm1d):
                nn.init.normal_(layer.weight, 1.0, 0.02)
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                #nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("linear"))
                layer.bias.data.zero_()
class MCDropout2d(nn.Dropout2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout2d(input, self.p, True, self.inplace)


class NPBlockRelu2d(nn.Module):
    """Block for Neural Processes."""

    def __init__(
        self, in_channels, out_channels, dropout=0, batchnorm=False, bias=False
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU()
        self.dropout2d = MCDropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else False

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # We pass a linear over it which operates on the Channels
        x = self.act(self.linear(x))

        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence, None) so we can use Dropout2d & BatchNorm2d
        x = x.permute(0, 2, 1)[:, :, :, None]

        if self.norm:
            x = self.norm(x)

        x = self.dropout2d(x)
        return x[:, :, :, 0].permute(0, 2, 1)

class BatchMLP(nn.Module):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).
    Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined
            in `basic.Linear`.
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_size
    """

    def __init__(
        self, input_size, output_size, num_layers=2, dropout=0, batchnorm=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.initial = NPBlockRelu2d(
            input_size, output_size, dropout=dropout, batchnorm=batchnorm
        )
        self.encoder = nn.Sequential(
            *[
                NPBlockRelu2d(
                    output_size, output_size, dropout=dropout, batchnorm=batchnorm
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.final = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        return self.final(x)


class LSTMBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout=0, batchnorm=False, bias=False, num_layers=1, bidirectional=True):
        super().__init__()
        self._lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=bidirectional,
                bias=bias
        )
        self.num_layers = num_layers
        self.n_dirs = 2 if bidirectional else 1
        self.fc_hid = nn.Linear(2*out_channels, out_channels)

        self.fc_out = nn.Linear(2*out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.hidden_size = out_channels

        #initialize different layers
        #init_weights(self._lstm)
        #init_weights(self.fc_hid)
        #init_weights(self.fc_out)

    def forward(self, x):
        #(B,T,D )
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        """
        batch_size = x.shape[0]
        total_length = x.shape[1]
        h0 = torch.autograd.Variable(torch.randn(self.num_layers* self.n_dirs, batch_size, self.hidden_size).type_as(x).to(x.device), requires_grad=True )
        c0 = torch.autograd.Variable(torch.randn(self.num_layers* self.n_dirs, batch_size, self.hidden_size).type_as(x).to(x.device), requires_grad=True )

        src_len = torch.LongTensor([torch.max((x[i,:, 0]!=0).nonzero()).item()+1 for i in range(x.shape[0])])

        #print(f"LSTMBlock input shape of the block {x.shape}, sequence length {src_len}")
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_len.cpu().numpy(), batch_first=True, enforce_sorted=False)

        assert not torch.isnan(packed_x.data).any()

        packed_outputs, hidden_state = self._lstm(packed_x, (h0,c0))

        hidden = hidden_state[0]


        hidden = hidden.view(self.num_layers, self.n_dirs, batch_size, self.hidden_size).sum(dim=1)
        # pad packed output sequence (B,T,2*H )
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.norm(torch.relu(self.fc_out(outputs))) #(B,:src_len, H)

        #hidden_state:(B, H)
        hidden_state = torch.tanh(self.fc_hid(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # Combine the forward and backward RNN's hidden states to be input to decoder

        return outputs, hidden_state
