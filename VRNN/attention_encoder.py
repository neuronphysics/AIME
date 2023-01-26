import torch
import torch.nn as nn
from Blocks import BatchMLP, LSTMBlock
from typing import Optional
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable, Function
#based on https://github.com/sarthmit/BRIMs/blob/f8af67e863ea751b45b70cc7a7b91fb277beb329/MNIST/attention.py
class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0
    
class Sparse_grad_attention(torch.autograd.Function):
    def __init__(self, top_k):
        super(Sparse_grad_attention,self).__init__()

        self.sa = Sparse_attention(top_k=top_k)

    def forward(self, inp): 
        
        sparsified = self.sa(inp)
        self.save_for_backward(inp, sparsified)

        return inp

    def backward(self, grad_output):
        inp, sparsified = self.saved_tensors
        #print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float()
    

class GroupLinearLayer(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)

class Sparse_attention(nn.Module):
    def __init__(self, top_k = 5):
        super(Sparse_attention,self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        #returns both value and location
        #attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            #delta = torch.min(attn_s, dim = 1)[0]
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements 
            #delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            #delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize
            delta = delta.reshape((delta.shape[0],1))


        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
        attn_w_sum = attn_w_sum + eps 
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        #print('attn', attn_w_normalize)

        return attn_w_normalize


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, topk, grad_sparse, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.grad_sparse = grad_sparse
        #print('top 2 sparsity')
        self.topk = topk
        self.sa = Sparse_attention(top_k=topk) #k=2
        

    def forward(self, q, k, v, mask=None):

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Forward of Scaled Dot Product Attention~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("q: ", q.size())
        # print("k: ", k.size())
        # print("v: ", v.size())
        # print("k transpose: ", k.transpose(1,2).size())
        # input()

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        #print('in forward attn shape', attn.shape)

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        #attn = self.dropout(attn)
        attn = self.softmax(attn)
        #if random.uniform(0,1) < 0.0001 or attn[0].max() > 0.8:
        #    print('attn0', attn[0])

        #sparse_attn = attn*0.0
        #sparse_attn[:,0,0] += 1.0
        #sparse_attn[:,1,1] += 1.0
        #sparse_attn[:,2,2] += 1.0
        #attn = sparse_attn*1.0


        use_sparse = True#False
        #use_sparse = False

        #if random.uniform(0,1) < 0.0001:
        #    print('pre sparsity attn', attn.shape, attn)

        if use_sparse:
            mb, ins, outs = attn.shape[0], attn.shape[1], attn.shape[2]
            sparse_attn = attn.reshape((mb*ins, outs))
            #print('sparse attn shape 1', sparse_attn.shape)
            #sga = Sparse_grad_attention(2)
            if self.grad_sparse:
                sga = Sparse_grad_attention(self.topk)
                sparse_attn = sga(sparse_attn)
            else:
                sparse_attn = self.sa(sparse_attn)
            sparse_attn = sparse_attn.reshape((mb,ins,outs))
            attn = sparse_attn*1.0

        #print('post sparse', attn.shape)
        #print('post sparse', attn)

        #print('attention 0', attn[0])

        #attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class LayerConnAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_out, topk, grad_sparse, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), topk=topk, grad_sparse=grad_sparse)
        self.layer_norm = nn.LayerNorm(d_model)

        self.gate_fc = nn.Linear(n_head * d_v, d_out)
        self.fc = nn.Linear(n_head * d_v, d_out)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        #print('attn input shape', q.shape)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        #v = v.view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(sz_b*self.n_head, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(sz_b*self.n_head, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(sz_b*self.n_head, len_v, d_v) # (n*b) x lv x dv

        #mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=None)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        #print('output shape before fc', output.shape)

        #TODO: probably shouldn't just apply residual layer in the forward pass.

        output_init = output*1.0

        output = self.dropout(self.fc(output_init))
        #output = self.dropout(output_init)

        gate = F.sigmoid(self.gate_fc(output_init))

        #output = self.layer_norm(gate * output + (1 - gate) * residual)
        #output = gate * output + (1 - gate) * residual

        #output = residual + gate * F.tanh(output)
        output = gate * torch.tanh(output)
        #output

        #print('attn', attn[0])
        #print('output input diff', output - residual)

        return output, attn



############################################
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
        d_k = 32,
        d_v = 32,
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
            self._W = ScaledDotProductAttention(temperature=np.power(hidden_dim, 0.5), topk=n_heads, grad_sparse=False)
            self.n_heads = n_heads
            self.hidden_dim = hidden_dim
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W = LayerConnAttention(n_head=n_heads, d_model=hidden_dim, d_k=d_k, d_v=d_v, d_out=hidden_dim, topk=n_heads, grad_sparse=False)
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
        sz_b, len_q, _ = q.size()
        output, attn = self._W(q, k, v, mask=attention_mask)

        return output

    def _multihead_attention(self, k:torch.Tensor, v:torch.Tensor, q:torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        
        outs, attn = self._W( q, k, v, mask=attention_mask)
        
        return outs

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
