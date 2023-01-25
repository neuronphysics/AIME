import torch
import torch.nn as nn
from Blocks import BatchMLP, LSTMBlock
from typing import Optional
class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        torch.nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        attention_type,
        attention_layers=2,
        n_heads=8,
        x_dim=1,
        rep="mlp",
        dropout=0,
        batchnorm=False,
    ):
        super().__init__()
        self._rep = rep

        if self._rep == "mlp":
            self.mlp_k = BatchMLP(
                x_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )
            self.mlp_q = BatchMLP(
                x_dim,
                hidden_dim,
                attention_layers,
                dropout=dropout,
                batchnorm=batchnorm,
            )
        elif self._rep == "lstm":
            self._lstm = LSTMBlock(x_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=attention_layers)

        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_v = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_q = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W = AttnLinear(n_heads * hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        elif attention_type == "ptmultihead":
            self._W = torch.nn.MultiheadAttention(
                hidden_dim, n_heads, bias=True, dropout=dropout, batch_first=True,
            )
            self._attention_func = self._pytorch_multihead_attention
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        #k: (B,T,D_k)
        if self._rep == "mlp":
            k = self.mlp_k(k)
            q = self.mlp_q(q)
        elif self._rep == "lstm":
            k, _ = self._lstm(k)
            q, _ = self._lstm(q)
        rep = self._attention_func(k, v, q)
        return rep

    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)#(B, 1, D_v)
        rep = rep.repeat(1, total_points, 1) #(B, T, D_v)
        return rep

    def _laplace_attention(self, k, v, q, scale=0.5, normalise=True):
        # TODO: re-implement in the future as this runs too slowly.

        keys = torch.unsqueeze(keys, dim=1)  # [batch_size, 1, N_context, key_size]
        queries = torch.unsqueeze(queries, dim=1)  # [batch_size, N_target, 1, key_size]

        unnorm_weights = -torch.abs((keys - queries)/scale)  # [batch_size, N_target, N_context, key_size]
        unnorm_weights = torch.sum(unnorm_weights, dim=-1, keepdim=False) # [batch_size, N_target, N_context]

        if normalise:
            attention = torch.softmax(unnorm_weights, dim=-1)  # [batch_size, N_target, N_context]
        else:
            attention = 1 + torch.tanh(unnorm_weights)  # [batch_size, N_target, N_context]

        # Einstein summation over weights and values
        output= torch.matmul(attention, values)  # [batch_size, N_target, value_size]

        return output

    def _dot_attention(self, k:torch.Tensor, v:torch.Tensor, q:torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        scale = q.shape[-1] ** 0.5
        unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
        if attention_mask is not None:
            unnorm_weights = unnorm_weights.masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(unnorm_weights, dim=-1)
        # results: (B,T,D_v)
        rep = torch.einsum("bik,bkj->bij", weights, v)

        return rep

    def _multihead_attention(self, k:torch.Tensor, v:torch.Tensor, q:torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_, attention_mask)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

    def _pytorch_multihead_attention(self, k:torch.Tensor, v:torch.Tensor, q:torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        #query is ( T, B, D) when batch_first=False
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        o = self._W(q, k, v, attention_mask )[0]
        return o.permute(1, 0, 2)


class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=32,
        self_attention_type="dot",
        n_encoder_layers=3,
        batchnorm=False,
        dropout=0,
        attention_dropout=0,
        attention_layers=2,
        use_lstm=True
        ):
        super().__init__()
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_lstm:
            #https://github.com/alisafaya/shalstm/blob/fbce6414684dfe70e236e414bdcd7a113e20153a/shalstm/model/lstmblock.py
            self._encoder = LSTMBlock(input_dim, hidden_dim,  dropout=dropout, batchnorm=batchnorm, num_layers=n_encoder_layers)

        else:
            self._encoder = BatchMLP(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_encoder_layers)

        self._self_attention = Attention(
                                        hidden_dim,
                                        self_attention_type,
                                        attention_layers,
                                        x_dim=hidden_dim,
                                        rep="lstm",
                                        dropout=attention_dropout,
                                       )
        self._penultimate_layer = nn.Linear(2*hidden_dim, hidden_dim)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)

        self._use_lstm = use_lstm


    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)

        #encoder inputs[B, T, D_in+D_out]

        # Pass final axis through MLP
        if self._use_lstm:
            encoded, _ = self._encoder(encoder_input)
        else:
            encoded = self._encoder(encoder_input)

        # Aggregator: take the mean over all points
        attention_output = self._self_attention(encoded, encoded, encoded)
        #attention output torch.Size([B, :src_len, H])
        hidden = attention_output.mean(dim=1, keepdim=True)

        hidden = hidden.repeat(1, attention_output.shape[1],1)

        mean_attention =torch.cat([attention_output, hidden], dim=-1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        mean_repr = torch.relu(self._penultimate_layer(mean_attention))

        # Then apply further linear layers to output latent mu and log sigma
        mean    =  self._mean(mean_repr)
        logvar  =  nn.Softplus()(self._log_var(mean_repr))


        return mean, logvar
