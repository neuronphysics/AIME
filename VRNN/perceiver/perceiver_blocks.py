from typing import Dict, List, Mapping, Optional, Tuple
import torch.nn.functional as func
from VRNN.perceiver.perceiver_helpers import *

BATCH_DIM = 0
GROUPS_DIM = 1
INDEX_DIM = -2
CHANNELS_DIM = -1

RECONSTRUCTION_HEAD_NAME = 'reconstruction_head'


def regroup(inputs: torch.Tensor,
            num_output_groups: int,
            regroup_type: str) -> torch.Tensor:
    """Re-group an input array from [B, G, N, C] to [B, G', N', C].

    Args:
        inputs: the array to regroup.
        num_output_groups: The number of output groups G'.
        regroup_type: The regrouping strategy to use.
    Returns:
        The re-grouped array.
    """
    batch_size, num_input_groups, num_input_latents, num_channels = inputs.shape

    if regroup_type in ['reshape', 'transpose_reshape']:
        new_index_dim = num_input_groups * num_input_latents // num_output_groups
        if regroup_type == 'transpose_reshape':
            # [B, G, N, C] -> [B, N, G, C]
            # This leads to mixing between all input groups, rather than preferential
            # mixing between neighboring groups.
            inputs = inputs.transpose(1, 2)
        outputs = inputs.reshape(batch_size, num_output_groups, new_index_dim, num_channels)
    else:
        raise ValueError(f'Unknown regroup_type: {regroup_type}.')

    return outputs


class HiPCrossAttention(nn.Module):
    """
    A HiPCrossAttention module, including a dense block.

      Maps batched, grouped arrays of shape B x G x M x C to arrays of shape
      B x G x N x D.
    """

    def __init__(self,
                 input_num_channel: int,
                 num_groups: int,
                 drop_probs: float,
                 output_num_channels: int,
                 output_index_dim_eval: int,
                 output_index_dim_train: Optional[int] = None,
                 activation_name: str = 'sq_relu',
                 widening_factor: int = 1,
                 num_heads: int = 8,
                 use_post_attention_residual: bool = False):

        super().__init__()
        """
        Constructs a new HiPCrossAttention.

        Args:
            output_index_dim_train: The output index dimension size at train. Ignored
              if `query_inputs` is specified directly at call.
            output_index_dim_eval: The output index dimension size at eval. Ignored
              if `query_inputs` is specified directly at call.
            output_num_channels: The number of output channels.
            activation_name: The activation to use.
            widening_factor: The widening factor to use in the output MLP.
            num_heads: The number of heads to use in cross-attention.
            use_post_attention_residual: Enable the post-attention residual
              connection? This residual adds the query inputs to the output of
              attention.

        """
        self.output_index_dim_train = output_index_dim_train
        self.output_index_dim_eval = output_index_dim_eval

        self.output_num_channels = output_num_channels
        self.widening_factor = widening_factor

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.use_post_attention_residual = use_post_attention_residual

        # Initialize modules
        self.trainable_position_encoding = TrainablePositionEncoding(
            index_dim=num_groups * output_index_dim_eval,
            num_channels=output_num_channels
        )
        self.pre_attention_residual = conv_1d(
            input_channels=output_num_channels,
            output_channels=output_num_channels
        )
        self.dense_module = Dense(
            input_channels=output_num_channels,
            widening_factor=widening_factor,
            activation_name=activation_name,
            dropout_prob=drop_probs
        )
        self.attention_module = Attention(
            input_q_channel=output_num_channels,
            input_k_channel=input_num_channel,
            input_v_channel=input_num_channel,
            num_heads=num_heads,
            qk_channels=input_num_channel,
            v_channels=input_num_channel,
            output_channels=output_num_channels,
        )
        self.q_layer_norm = nn.LayerNorm(output_num_channels, elementwise_affine=True)
        self.kv_layer_norm = nn.LayerNorm(input_num_channel, elementwise_affine=True)
        self.dense_layer_norm = nn.LayerNorm(output_num_channels, elementwise_affine=True)

    def _subsample_query_inputs(self, query_inputs):
        """Randomly subsample the number of query inputs."""
        batch_size, num_groups, _, _ = query_inputs.shape
        weight_indices = []
        for _ in range(batch_size * num_groups):
            indices = torch.randperm(self.output_index_dim_eval)[:self.output_index_dim_train]
            weight_indices.append(indices)
        weight_indices = torch.stack(weight_indices).view(batch_size, num_groups, -1)
        one_hot_indices = func.one_hot(weight_indices, num_classes=self.output_index_dim_eval)
        query_inputs = torch.einsum('bgMc,bgmM->bgmc', query_inputs, one_hot_indices.to(query_inputs.dtype))
        return query_inputs

    def forward(self, inputs, is_training: bool, query_inputs=None, pre_attention_residual=None,
                attention_mask=None):
        batch_size, num_groups, _, _ = inputs.shape

        if query_inputs is None:
            assert self.output_index_dim_train is not None
            assert self.output_index_dim_eval is not None
            assert self.output_index_dim_eval >= self.output_index_dim_train
            assert self.output_num_channels is not None

            # Define index_dim dynamically based on inputs
            # index_dim = num_groups * self.output_index_dim_eval
            # self.trainable_position_encoding.update_index_dim(index_dim)

            query_inputs = self.trainable_position_encoding(batch_size=batch_size)
            # query_inputs = einshape('b(gm)c->bgmc', query_inputs, g=num_groups)

            b, gm, c = query_inputs.shape
            m = gm // num_groups
            query_inputs = torch.reshape(query_inputs, (b, num_groups, m, c))

            if is_training and self.output_index_dim_train < self.output_index_dim_eval:
                query_inputs = self._subsample_query_inputs(query_inputs)

        output_index_dim = query_inputs.shape[-2]
        output_num_channels = query_inputs.shape[-1]

        if pre_attention_residual is not None:
            assert pre_attention_residual.shape[-2] == output_index_dim
            residual_num_channels = pre_attention_residual.shape[-1]
            if residual_num_channels != output_num_channels:
                pre_attention_residual = self.pre_attention_residual(pre_attention_residual)
            query_inputs = query_inputs + pre_attention_residual

        # Define attention module channels dynamically
        self.attention_module.output_index_dim = output_index_dim
        # qk_channels = v_channels = input_k = input_v = inputs.shape[-1]
        # input_q = output_channels = query_inputs.shape[-1]
        # self.attention_module.update_params(qk_channels, v_channels, output_channels, input_q, input_k, input_v)

        attention = self.attention_module(
            inputs_q=self.q_layer_norm(query_inputs),
            inputs_kv=self.kv_layer_norm(inputs),
            attention_mask=attention_mask
        )

        if self.use_post_attention_residual:
            attention = attention + query_inputs

        output = attention + self.dense_module(
            self.dense_layer_norm(attention), is_training=is_training
        )

        return output


class SelfAttention(nn.Module):
    """
    A self-attention module, including a dense block in PyTorch.
    """

    def __init__(self,
                 input_channels: int,
                 widening_factor: int = 4,
                 dropout_prob: float = 0.0,
                 dropout_attn_prob: float = 0.0,
                 drop_path_rate: float = 0.0,
                 num_heads: int = 8,
                 att_init_scale: float = 1.0,
                 dense_init_scale: float = 1.0,
                 qk_channels: Optional[int] = None,
                 v_channels: Optional[int] = None,
                 activation_name: str = 'sq_relu'):
        super(SelfAttention, self).__init__()
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._qk_channels = qk_channels if qk_channels is not None else input_channels
        self._v_channels = v_channels if v_channels is not None else self._qk_channels
        self.drop_path_rate = drop_path_rate

        if drop_path_rate > 0.:
            self._drop_path = StochasticDepth(drop_path_rate)
        else:
            self._drop_path = nn.Identity()

        self.layer_norm1 = nn.LayerNorm(input_channels)
        self.attention = Attention(
            input_q_channel=input_channels,
            input_k_channel=input_channels,
            input_v_channel=input_channels,
            num_heads=num_heads,
            init_scale=att_init_scale,
            dropout_prob=dropout_attn_prob,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            output_channels=input_channels)

        self.layer_norm2 = nn.LayerNorm(input_channels)
        self.dense = Dense(
            input_channels=input_channels,
            widening_factor=self._widening_factor,
            dropout_prob=self._dropout_prob,
            init_scale=self._dense_init_scale,
            activation_name=activation_name)

    def forward(self, inputs: torch.Tensor, is_training: bool,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = inputs

        # Self-attention block
        qkv_inputs = self.layer_norm1(inputs)
        self.attention.output_index_dim = qkv_inputs.shape[-2]
        attention_out = self.attention(qkv_inputs, qkv_inputs, attention_mask)
        if self.drop_path_rate > 0.:
            x = x + self._drop_path(attention_out, is_training)
        else:
            x = x + self._drop_path(attention_out)

        # Dense block
        dense_out = self.dense(self.layer_norm2(x), is_training)

        if self.drop_path_rate > 0.:
            x = x + self._drop_path(dense_out, is_training)
        else:
            x = x + self._drop_path(dense_out)

        return x


class PerceiverBlock(nn.Module):
    """
    The PerceiverBlock combines regrouping, cross- and self-attention in PyTorch.
    """

    def __init__(self,
                 input_num_channel: int,
                 num_output_groups: int,
                 output_index_dim: int,
                 num_output_channels: int,
                 num_self_attend_layers: int,
                 num_self_attend_heads: int,
                 self_attend_widening_factor: int = 4,
                 num_cross_attend_heads: int = 1,
                 cross_attend_widening_factor: int = 1,
                 regroup_inputs: bool = True,
                 regroup_type: str = 'reshape',
                 activation_name: str = 'sq_relu',
                 use_post_attention_residual: bool = True,
                 output_index_dim_train: Optional[int] = None,
                 output_index_dim_eval: Optional[int] = None,
                 dropout_prob: float = 0.0,
                 drop_path_rate: float = 0.0):
        super(PerceiverBlock, self).__init__()

        self.num_output_groups = num_output_groups
        self.num_output_channels = num_output_channels
        self.regroup_inputs = regroup_inputs
        self.regroup_type = regroup_type
        self.output_index_dim_train = output_index_dim_train or output_index_dim
        self.output_index_dim_eval = output_index_dim_eval or output_index_dim

        self.projector = HiPCrossAttention(
            input_num_channel=input_num_channel, num_groups=num_output_groups,
            output_index_dim_train=self.output_index_dim_train,
            output_index_dim_eval=self.output_index_dim_eval,
            output_num_channels=num_output_channels, activation_name=activation_name,
            widening_factor=cross_attend_widening_factor,
            num_heads=num_cross_attend_heads,
            use_post_attention_residual=use_post_attention_residual, drop_probs=dropout_prob)

        self.self_attentions = nn.ModuleList([
            SelfAttention(
                input_channels=num_output_channels,
                widening_factor=self_attend_widening_factor,
                dropout_prob=dropout_prob,
                drop_path_rate=drop_path_rate,
                num_heads=num_self_attend_heads,
                att_init_scale=1.0,  # Initialize as desired
                dense_init_scale=1.0,  # Initialize as desired
                activation_name=activation_name
            ) for _ in range(num_self_attend_layers)
        ])

    def forward(self, inputs: torch.Tensor,
                is_training: bool,
                pre_attention_residual: Optional[Dict[str, torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # batch_size = inputs.shape[0]
        # output_index_dim = self.output_index_dim_train if is_training else self.output_index_dim_eval
        # (batch, groups, index, channels)
        assert len(inputs.shape) == 4
        if self.regroup_inputs:
            inputs = regroup(inputs, self.num_output_groups, self.regroup_type)
        else:
            assert inputs.shape[1] == self.num_output_groups, "Number of input and output groups must match."

        z = self.projector(inputs, pre_attention_residual=pre_attention_residual,
                           is_training=is_training, attention_mask=attention_mask)

        for self_attend in self.self_attentions:
            z = self_attend(z, is_training=is_training)

        return z


class Embedder(nn.Module):
    """
    Projects inputs to the target number of channels.

    Inputs should be a dictionary of {modality_name:
    (batch_size, index_dim, num_channels)}. The output format will be similar, but
    with the new number of channels.

    Note both inputs and outputs are ungrouped. Grouping is handled by the
    Grouper module.
    """

    def __init__(self,
                 modalities: Mapping[str, torch.Tensor],
                 num_embedding_channels: int,
                 with_bias: bool = True):
        super().__init__()
        self.num_embedding_channels = num_embedding_channels
        self.modalities = modalities

        # Create a dictionary of 1D convolutional layers for each modality
        self.embed_layers = nn.ModuleDict({
            modality: nn.Linear(in_features=data.shape[CHANNELS_DIM],
                                out_features=num_embedding_channels,
                                bias=with_bias)
            for modality, data in modalities.items()
        })

        self.un_embed_layers = nn.ModuleDict({
            modality: nn.Linear(in_features=num_embedding_channels,
                                out_features=data.shape[CHANNELS_DIM],
                                bias=with_bias)
            for modality, data in modalities.items()
        })
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, inputs: Mapping[str, torch.Tensor], un_embed: bool = False) -> Dict[str, torch.Tensor]:
        """
        Takes raw inputs and embeds them to num_embedding_channels.

        Args:
          inputs: A dictionary of modality name and (batch, index, channels) value.
          un_embed: determine if we should do un-embed

        Returns:
          A dictionary of modality name and (batch, index, num_embedding_channels).
        --------------
        Reverses an embed operation, reproducing the shape of original inputs.

        Args:
          inputs: A dictionary of modality name and (batch, index,
            num_embedding_channels).

        Returns:
          A dictionary of modality name and (batch, index, num_embedding_channels).

        """
        if un_embed:
            assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
        else:
            assert_input_shapes(inputs, expected_rank=3)
        out = {}
        layers = self.un_embed_layers if un_embed else self.embed_layers

        for modality_name, value in inputs.items():
            layer = layers[modality_name]
            out[modality_name] = layer(value)

        return out


class PositionEncoder(nn.Module):
    """
    Adds position encodings to input channels.

        Inputs should be a dictionary of {modality_name:
        (batch_size, index_dim, num_channels)}. The output format will be identical.

        Note both inputs and outputs are ungrouped. Grouping is handled by the
        Grouper module.
    """

    def __init__(self, modalities: Mapping[str, torch.Tensor], embed_out_channel: int,
                 num_position_encoding_channels: Optional[int] = None):
        super().__init__()
        self.num_position_encoding_channels = num_position_encoding_channels
        self.position_encodings = nn.ModuleDict()

        # Creating a dictionary of position encoders for each modality
        for modality, val in modalities.items():
            self.position_encodings[modality] = TrainablePositionEncoding(
                index_dim=val.shape[INDEX_DIM],
                num_channels=embed_out_channel
            )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Adds position encodings to the inputs and also returns the encodings.

        Args:
          inputs: A dictionary of {modality_name:
           ( batch_size, index_dim, num_channels)} inputs.

        Returns:
          A tuple of the inputs with added position encodings, as well as the
          raw encodings.
        """
        assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
        out = {}
        pos_encodings = {}

        for modality_name, value in inputs.items():
            batch_size, index_dim, num_channels = value.shape
            pos_encodings_i = self.position_encodings[modality_name](batch_size)

            pos_encodings[modality_name] = pos_encodings_i
            out[modality_name] = value + pos_encodings[modality_name]

        return out, pos_encodings


def eigenshape(tensor, group_idx_length):
    """
    Reshape a tensor from 'b(gm)...' to 'bgm...'.

    Args:
    - tensor: The input tensor with shape (b, gm, ...)
    - group_idx_length: Length of the group index 'g'

    Returns:
    - reshaped_tensor: The tensor reshaped to (b, g, m, ...)
    """
    batch_size, combined_dim = tensor.shape[:2]
    other_dims = tensor.shape[2:]
    m_dim = combined_dim // group_idx_length

    # Reshape while keeping the batch size and other dimensions unchanged
    reshaped_tensor = tensor.view(batch_size, group_idx_length, m_dim, *other_dims)

    return reshaped_tensor


class ConstNumGrouper(nn.Module):
    """
    Groups inputs into a constant number of groups.

    The group size will grow based on the inputs.

    Inputs should be a dictionary of {modality_name:
    (batch_size, index_dim, num_channels)}. The output format will be (batch_size,
    num_groups, new_index_dim (computed), num_channels).

    Notes: Inputs will be ordered based on the insertion order of the dict. Make
    sure this is consistent across calls. batch_size and num_channels must be
    constant across modalities.

    The Grouper will ensure that multiple modalities are not mixed in a single
    group. Padding will be added proportionally at the end of each group, with
    extra padding at the last group of each modality.
    """

    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self._group_map = None

    def _build_group_map(self, inputs: Dict[str, torch.Tensor]):
        index_dims = [v.shape[1] for v in inputs.values()]
        num_groups_per_modality, index_dim_per_group = assign_groups_to_modalities(
            self.num_groups, index_dims)

        group_map = []
        next_group_id = 0
        for (name, value), num_modality_groups in zip(inputs.items(), num_groups_per_modality):
            index_dim = value.shape[1]
            assigned_groups = list(range(next_group_id, next_group_id + num_modality_groups))
            next_group_id += num_modality_groups

            final_padding = padding_to_make_divisible(index_dim, num_modality_groups)
            local_index_dim_per_group = (index_dim + final_padding) // num_modality_groups
            group_padding = index_dim_per_group - local_index_dim_per_group

            group_map.append({
                "modality_name": name,
                "group_idx": assigned_groups,
                "final_padding": final_padding,
                "group_padding": group_padding
            })

        self._group_map = group_map

    def group(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Groups a given input with the appropriate padding.

        This method can be called multiple times on inputs that require similar
        grouping and padding (e.g., a sample and its attention mask).

        Args:
        inputs: A dict of modality names and (batch, index, channel) values.

        Returns:
        A tensor of shape (batch, group, index, channel).
        """
        assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
        self._build_group_map(inputs)

        grouped_inputs = []
        for group_info, value in zip(self._group_map, inputs.values()):
            x = torch.nn.functional.pad(value, (0, 0, 0, group_info['final_padding']))
            # Reshaping and permuting to group
            x = eigenshape(x, len(group_info['group_idx']))

            padding_num_left = group_info['group_padding']
            padding_len = group_info['group_padding'] if group_info['group_padding'] < x.shape[-2] else x.shape[-2] - 1
            m = torch.nn.ReflectionPad2d((0, 0, 0, padding_len))
            while padding_num_left > padding_len:
                x = m(x)
                padding_num_left -= padding_len
            if padding_num_left != 0:
                n = torch.nn.ReflectionPad2d((0, 0, 0, padding_num_left))
                x = n(x)
            grouped_inputs.append(x)

        return torch.cat(grouped_inputs, dim=1)

    def ungroup(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert len(latents.shape) == 4
        out = {}

        for group_info in self._group_map:
            x = latents[:, group_info['group_idx'], :, :]
            x = x[:, :, :x.shape[2] - group_info['group_padding'], :]
            b, g, m, c = x.shape
            x = torch.reshape(x, (b, g * m, c))
            # Remove final padding.
            x = x[:, :x.shape[1] - group_info['final_padding'], :]

            out[group_info['modality_name']] = x

        return out


class ConcatenateGrouper(nn.Module):
    """
    Concatenates inputs into a single group, shared across all modalities.

    Inputs should be a dictionary of {modality_name:
    (batch_size, index_dim, num_channels)}. The output format will be (batch_size,
    num_groups, total_index_dim, num_channels), where
    total_index_dim = sum_{modality_i} index_dim_i.

    Notes: Inputs will be ordered based on the insertion order of the dict. Make
    sure this is consistent across calls. batch_size and num_channels must be
    constant across modalities
    """

    def __init__(self):
        super().__init__()
        self._index_dims = None
        self._input_names = None

    def group(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Groups a given input.

        Args:
            inputs: A dict of modality names and (batch, index, channel) values.

        Returns:
            A tensor of shape (batch, group, index, channel).
        """
        assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
        if not self._index_dims or not self._input_names:
            self._input_names = list(inputs.keys())
            self._index_dims = [v.shape[1] for v in inputs.values()]

        # Concatenate along the index dimension [B, (I_0 + I_1 + ... + I_N), C]
        grouped = torch.cat(list(inputs.values()), dim=1)
        # Add a dummy group axis
        grouped = grouped.unsqueeze(1)

        return grouped

    def ungroup(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Ungroups a given input into a dict of modalities and values.

        Args:
            latents: A tensor of (batch, group, index, channel).

        Returns:
            A dict of the original modality names and their values.
        """
        assert latents.dim() == 4, "Latents tensor should be 4-dimensional"

        start_idx = 0
        out = {}
        for name, index_dim in zip(self._input_names, self._index_dims):
            end_idx = start_idx + index_dim
            # [B, 1, (I_0 + I_1 + ... + I_N), C] -> [B, I_i, C]
            out[name] = latents[:, 0, start_idx:end_idx, :]
            start_idx = end_idx

        return out


class ReconstructionHead(nn.Module):
    """
    Produces a reconstruction from perceiver latents and an MAE query.
    """

    def __init__(self, num_groups: int, input_num_channel: int, output_num_channels: int, output_index_dim_eval: int,
                 drop_probs: float, use_post_attention_residual: bool = False):
        super().__init__()
        self.use_post_attention_residual = use_post_attention_residual
        # Initialize the HiPCrossAttention module here
        self.projector = HiPCrossAttention(input_num_channel=input_num_channel, num_groups=num_groups,
                                           output_num_channels=output_num_channels,
                                           output_index_dim_eval=output_index_dim_eval, widening_factor=1, num_heads=1,
                                           use_post_attention_residual=self.use_post_attention_residual,
                                           drop_probs=drop_probs)

    def forward(self,
                latents: torch.Tensor,
                mae_query: torch.Tensor,
                is_training: bool) -> torch.Tensor:
        """
        Given latents and an MAE query, builds the reconstruction.

        Args:
            latents: The output of a PerceiverBlock.
            mae_query: MAE query - typically the output of a PositionEncoder.
            is_training: Boolean flag indicating training mode.

        Returns:
            A tensor representing the grouped array of reconstructions for the query.
        """
        assert latents.dim() == 4, "Latents tensor should be 4-dimensional"
        assert mae_query.dim() == 4, "MAE query tensor should be 4-dimensional"

        predictions = self.projector(inputs=latents, query_inputs=mae_query, is_training=is_training)

        return predictions


def assert_input_shapes(inputs: Mapping[str, torch.Tensor],
                        expected_rank: int,
                        constant_channels: bool = False):
    """
    Given an inputs dictionary, asserts all shapes are correct.

    Args:
        inputs: A dictionary of tensors with modality names as keys.
        expected_rank: Expected number of dimensions in each tensor.
        constant_channels: If True, asserts that the number of channels is the same for all inputs.

    Returns:
        batch_size: The common batch size across all inputs.
        num_channels: The common number of channels, if constant_channels is True; otherwise, None.
    """
    batch_size = None
    num_channels = None

    for modality_name, values in inputs.items():
        assert len(values.shape) == expected_rank, f"Shape rank of input '{modality_name}' is not as expected."

        if batch_size is None:
            batch_size = values.shape[0]  # Assuming batch size is the first dimension
            num_channels = values.shape[-1]  # Assuming channels are the last dimension
        else:
            assert batch_size == values.shape[0], f"Batch size is inconsistent for input '{modality_name}'."
            if constant_channels:
                assert num_channels == values.shape[
                    -1], f"Number of channels is inconsistent for input '{modality_name}'."

    return batch_size, num_channels if constant_channels else None
