import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Iterable, Any, Dict, Mapping
from torch import Tensor
from einops import reduce
import functools, math
import pathlib
from torch.utils.checkpoint import checkpoint as ckpt
def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj

DEFAULT_WEIGHT_INIT = "default"
def init_parameters(layers: Union[nn.Module, Iterable[nn.Module]], weight_init: str = "default"):
    assert weight_init in ("default", "he_uniform", "he_normal", "xavier_uniform", "xavier_normal")
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)

        if hasattr(layer, "weight") and layer.weight is not None:
            gain = 1.0
            if isinstance(layers, nn.Sequential):
                if idx < len(layers) - 1:
                    next = layers[idx + 1]
                    if isinstance(next, nn.ReLU):
                        gain = 2**0.5

            if weight_init == "he_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, gain)
            elif weight_init == "he_normal":
                torch.nn.init.kaiming_normal_(layer.weight, gain)
            elif weight_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight, gain)
            elif weight_init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight, gain)

Shape = Tuple[int]

DType = Any
Array = torch.Tensor # np.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  


class MLP(nn.Module):
	"""Simple MLP with one hidden layer and optional pre-/post-layernorm."""

	def __init__(self,
				 input_size: int, # FIXME: added because or else can't instantiate submodules
				 hidden_size: int,
				 output_size: int, # if not given, should be inputs.shape[-1] at forward
				 num_hidden_layers: int = 1,
				 activation_fn: nn.Module = nn.ReLU,
				 layernorm: Optional[str] = None,
				 activate_output: bool = False,
				 residual: bool = False,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_hidden_layers = num_hidden_layers
		self.activation_fn = activation_fn
		self.layernorm = layernorm
		self.activate_output = activate_output
		self.residual = residual
		self.weight_init = weight_init

		# submodules
		## layernorm
		if self.layernorm == "pre":
			self.layernorm_module = nn.LayerNorm(input_size, eps=1e-6)
		elif self.layernorm == "post":
			self.layernorm_module = nn.LayerNorm(output_size, eps=1e-6)
		## mlp
		self.model = nn.ModuleList()
		self.model.add_module("dense_mlp_0", nn.Linear(self.input_size, self.hidden_size))
		self.model.add_module("dense_mlp_0_act", self.activation_fn())
		for i in range(1, self.num_hidden_layers):
			self.model.add_module(f"den_mlp_{i}", nn.Linear(self.hidden_size, self.hidden_size))
			self.model.add_module(f"dense_mlp_{i}_act", self.activation_fn())
		self.model.add_module(f"dense_mlp_{self.num_hidden_layers}", nn.Linear(self.hidden_size, self.output_size))
		if self.activate_output:
			self.model.add_module(f"dense_mlp_{self.num_hidden_layers}_act", self.activation_fn())
		for name, module in self.model.named_children():
			if 'act' not in name:
				# nn.init.xavier_uniform_(module.weight)
				init_fn[weight_init['linear_w']](module.weight)
				init_fn[weight_init['linear_b']](module.bias)

	def forward(self, inputs: Array) -> Array:

		x = inputs
		if self.layernorm == "pre":
			x = self.layernorm_module(x)
		for layer in self.model:
			x = layer(x)
		if self.residual:
			x = x + inputs
		if self.layernorm == "post":
			x = self.layernorm_module(x)
		return x

class myGRUCell(nn.Module):
	"""GRU cell as nn.Module

	Added because nn.GRUCell doesn't match up with jax's GRUCell...
	This one is designed to match ! (almost; output returns only once)

	The mathematical definition of the cell is as follows

  	.. math::

		\begin{array}{ll}
		r = \sigma(W_{ir} x + W_{hr} h + b_{hr}) \\
		z = \sigma(W_{iz} x + W_{hz} h + b_{hz}) \\
		n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
		h' = (1 - z) * n + z * h \\
		\end{array}
	"""

	def __init__(self,
				 input_size: int,
				 hidden_size: int,
				 gate_fn = torch.sigmoid,
				 activation_fn = torch.tanh,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.gate_fn = gate_fn
		self.activation_fn = activation_fn
		self.weight_init = weight_init

		# submodules
		self.dense_ir = nn.Linear(input_size, hidden_size)
		self.dense_iz = nn.Linear(input_size, hidden_size)
		self.dense_in = nn.Linear(input_size, hidden_size)
		self.dense_hr = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hz = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hn = nn.Linear(hidden_size, hidden_size)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		recurrent_weight_init = nn.init.orthogonal_
		if self.weight_init is not None:
			weight_init = init_fn[self.weight_init['linear_w']]
			bias_init = init_fn[self.weight_init['linear_b']]
		else:
			# weight init not given
			stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
			weight_init = bias_init = lambda weight: nn.init.uniform_(weight, -stdv, stdv)
		# input weights
		weight_init(self.dense_ir.weight)
		bias_init(self.dense_ir.bias)
		weight_init(self.dense_iz.weight)
		bias_init(self.dense_iz.bias)
		weight_init(self.dense_in.weight)
		bias_init(self.dense_in.bias)
		# hidden weights
		recurrent_weight_init(self.dense_hr.weight)
		recurrent_weight_init(self.dense_hz.weight)
		recurrent_weight_init(self.dense_hn.weight)
		bias_init(self.dense_hn.bias)
	
	def forward(self, inputs, carry):
		h = carry
		# input and recurrent layeres are summed so only one needs a bias
		r = self.gate_fn(self.dense_ir(inputs) + self.dense_hr(h))
		z = self.gate_fn(self.dense_iz(inputs) + self.dense_hz(h))
		# add bias because the linear transformations aren't directly summed
		n = self.activation_fn(self.dense_in(inputs) +
							   r * self.dense_hn(h))
		new_h = (1. - z) * n + z * h
		return new_h


def lecun_uniform_(tensor, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    var = gain / float(fan_in)
    a = math.sqrt(3 * var)
    return nn.init._no_grad_uniform_(tensor, -a, a)


def lecun_normal_(tensor, gain=1., mode="fan_in"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale_mode = fan_in
    elif mode == "fan_out":
        scale_mode = fan_out
    else:
        raise NotImplementedError
    var = gain / float(scale_mode)
    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(var) / .87962566103423978
    # return nn.init._no_grad_normal_(tensor, 0., std)
    kernel = torch.nn.init._no_grad_trunc_normal_(tensor, 0, 1, -2, 2) * std
    with torch.no_grad():
        tensor[:] = kernel[:]
    return tensor

def lecun_normal_fan_out_(tensor, gain=1.):
    return lecun_normal_(tensor, gain=gain, mode="fan_out")

def lecun_normal_convtranspose_(tensor, gain=1.):
    # for some reason, the convtranspose weights are [in_channels, out_channels, kernel, kernel]
    # but the _calculate_fan_in_and_fan_out treats dim 1 as fan_in and dim 0 as fan_out.
    # so, for convolution weights, have to use fan_out instead of fan_in
    # which is actually using fan_in instead of fan_out
    return lecun_normal_fan_out_(tensor, gain=gain)

init_fn = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'lecun_uniform': lecun_uniform_,
    'lecun_normal': lecun_normal_,
    'lecun_normal_fan_out': lecun_normal_fan_out_,
    'ones': nn.init.ones_,
    'zeros': nn.init.zeros_,
    'default': lambda x: x}
def init_param(name, gain=1.):
    assert name in init_fn.keys(), "not a valid init method"
    # return init_fn[name](tensor, gain)
    return functools.partial(init_fn[name], gain=gain)




class SlotAttention(nn.Module):
    """Slot Attention module.

    Note: This module uses pre-normalization by default.
    """
    def __init__(self,
                 input_size: int, # size of encoded inputs. 
                 slot_size: int, # fixed size. or same as qkv_size.
                 qkv_size: int = None, # fixed size, or slot size. 
                 num_iterations: int = 1,
                 mlp_size: Optional[int] = None,
                 epsilon: float = 1e-8,
                 num_heads: int = 1,
                 weight_init: str = 'xavier_uniform'
                ):
        super().__init__()

        self.input_size = input_size
        self.slot_size = slot_size
        self.qkv_size = qkv_size if qkv_size is not None else slot_size
        self.num_iterations = num_iterations
        self.mlp_size = mlp_size
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.weight_init = weight_init
        # other definitions
        self.head_dim = self.qkv_size // self.num_heads

        # shared modules
        ## gru
        self.gru = myGRUCell(slot_size, slot_size, weight_init=weight_init)

        ## weights
        self.dense_q = nn.Linear(slot_size, self.qkv_size, bias=False)
        self.dense_k = nn.Linear(input_size, self.qkv_size, bias=False)
        self.dense_v = nn.Linear(input_size, self.qkv_size, bias=False)
        init_fn[weight_init['linear_w']](self.dense_q.weight)
        init_fn[weight_init['linear_w']](self.dense_k.weight)
        init_fn[weight_init['linear_w']](self.dense_v.weight)

        ## layernorms
        self.layernorm_q = nn.LayerNorm(slot_size, eps=1e-6)
        self.layernorm_input = nn.LayerNorm(input_size, eps=1e-6)

        ## attention
        self.inverted_attention = InvertedDotProductAttention(
            input_size=self.qkv_size, output_size=slot_size,
            num_heads=self.num_heads, norm_type="mean",
            epsilon=epsilon, weight_init=weight_init)
        
        ## output transform
        if self.mlp_size is not None:
            self.mlp = MLP(
                input_size=slot_size, hidden_size=self.mlp_size,
                output_size=slot_size, layernorm="pre", residual=True,
                weight_init=weight_init)

    def forward(self, slots: Array, inputs: Array) -> Array:
        """Slot Attention module forward pass."""
        B, O, D = slots.shape
        _, L, M = inputs.shape

        # inputs.shape = (b, n_inputs, input_size).
        inputs = self.layernorm_input(inputs)
        # k.shape = (b, n_inputs, num_heads, head_dim).
        k = self.dense_k(inputs).view(B, L, self.num_heads, self.head_dim)
        # v.shape = (b, n_inputs, num_heads, head_dim).
        v = self.dense_v(inputs).view(B, L, self.num_heads, self.head_dim)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):

            # Inverted dot-product attention.
            slots_n = self.layernorm_q(slots)
            ## q.shape = (b, num_objects, num_heads, qkv_size).
            q = self.dense_q(slots_n).view(B, O, self.num_heads, self.head_dim)
            updates, attn = self.inverted_attention(query=q, key=k, value=v)
            # attn: [B, H, S, L]
            attn_vis = attn.mean(dim=1)          # [B, S, L]

            # Recurrent update.
            slots = self.gru(
                updates.reshape(-1, D), 
                slots.reshape(-1, D))
            slots = slots.reshape(B, -1, D)

            # Feedforward block with pre-normalization.
            if self.mlp_size is not None:
                slots = self.mlp(slots)

        return {"slots":slots,  "masks":attn_vis}



class InvertedDotProductAttention(nn.Module):
    """Inverted version of dot-product attention (softmax over query axis)."""

    def __init__(self,
                 input_size: int, # qkv_size # FIXME: added for submodules
                 output_size: int, # FIXME: added for submodules
                 num_heads: Optional[int] = 1, # FIXME: added for submodules
                 norm_type: Optional[str] = "mean", # mean, layernorm, or None
                 # multi_head: bool = False, # FIXME: can infer from num_heads.
                 epsilon: float = 1e-8,
                 dtype: DType = torch.float32,
                 weight_init = None
                 # precision # not used
                ):
        super().__init__()

        assert num_heads >= 1 and isinstance(num_heads, int)

        self.input_size = input_size
        self.output_size = output_size
        self.norm_type = norm_type
        self.num_heads = num_heads
        self.multi_head = True if num_heads > 1 else False
        self.epsilon = epsilon
        self.dtype = dtype
        self.weight_init = weight_init
        # other definitions
        self.head_dim = input_size // self.num_heads

        # submodules
        self.attn_fn = GeneralizedDotProductAttention(
            inverted_attn=True,
            renormalize_keys=True if self.norm_type == "mean" else False,
            epsilon=self.epsilon,
            dtype=self.dtype)
        if self.multi_head:
            self.dense_o = nn.Linear(input_size, output_size, bias=False)
            init_fn[weight_init['linear_w']](self.dense_o.weight)
        if self.norm_type == "layernorm":
            self.layernorm = nn.LayerNorm(output_size, eps=1e-6)

    def forward(self, query: Array, key: Array, value: Array) -> Array:
        """Computes inverted dot-product attention.

        Args:
            qk_features = [num_heads, head_dim] = qkv_dim
            query: Queries with shape of `[batch, q_num, qk_features]`.
            key: Keys with shape of `[batch, kv_num, qk_features]`.
            value: Values with shape of `[batch, kv_num, v_features]`.
            train: Indicating whether we're training or evaluating.

        Returns:
            Output of shape `[batch, n_queries, v_features]`
        """
        B, Q = query.shape[:2]

        # Apply attention mechanism
        output, attn = self.attn_fn(query=query, key=key, value=value)

        if self.multi_head:
            # Multi-head aggregation. Equivalent to concat + dense layer.
            output = self.dense_o(output.view(B, Q, self.input_size)).view(B, Q, self.output_size)
        else:
            # Remove head dimension.
            output = output.squeeze(-2)

        if self.norm_type == "layernorm":
            output = self.layernorm(output)

        return output, attn


class GeneralizedDotProductAttention(nn.Module):
    """Multi-head dot-product attention with customizable normalization axis.

    This module supports logging of attention weights in a variable collection.
    """

    def __init__(self,
                 dtype: DType = torch.float32,
                 # precision: Optional[] # not used
                 epsilon: float = 1e-8,
                 inverted_attn: bool = False,
                 renormalize_keys: bool = False,
                 attn_weights_only: bool = False
                ):
        super().__init__()

        self.dtype = dtype
        self.epsilon = epsilon
        self.inverted_attn = inverted_attn
        self.renormalize_keys = renormalize_keys
        self.attn_weights_only = attn_weights_only

    def forward(self, query: Array, key: Array, value: Array,
                train: bool = False, **kwargs) -> Array:
        """Computes multi-head dot-product attention given query, key, and value.

        Args:
            query: Queries with shape of `[batch..., q_num, num_heads, qk_features]`.
            key: Keys with shape of `[batch..., kv_num, num_heads, qk_features]`.
            value: Values with shape of `[batch..., kv_num, num_heads, v_features]`.
            train: Indicating whether we're training or evaluating.
            **kwargs: Additional keyword arguments are required when used as attention
                function in nn.MultiHeadDotPRoductAttention, but they will be ignored here.

        Returns:
            Output of shape `[batch..., q_num, num_heads, v_features]`.
        """
        del train # Unused.

        assert query.ndim == key.ndim == value.ndim, (
            "Queries, keys, and values must have the same rank.")
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
            "Query, key, and value batch dimensions must match.")
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
            "Query, key, and value num_heads dimensions must match.")
        assert key.shape[-3] == value.shape[-3], (
            "Key and value cardinality dimensions must match.")
        assert query.shape[-1] == key.shape[-1], (
            "Query and key feature dimensions must match.")

        if kwargs.get("bias") is not None:
            raise NotImplementedError(
                "Support for masked attention is not yet implemented.")

        if "dropout_rate" in kwargs:
            if kwargs["dropout_rate"] > 0.:
                raise NotImplementedError("Support for dropout is not yet implemented.")

        # Temperature normalization.
        qk_features = query.shape[-1]
        query = query / (qk_features ** 0.5) # torch.sqrt(qk_features)

        # attn.shape = (batch..., num_heads, q_num, kv_num)
        attn = torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)) # bhqd @ bhdk -> bhqk

        if self.inverted_attn:
            attention_dim = -2 # Query dim
        else:
            attention_dim = -1 # Key dim

        # Softmax normalization (by default over key dim)
        attn = torch.softmax(attn, dim=attention_dim, dtype=self.dtype)

        if self.renormalize_keys:
            # Corresponds to value aggregation via weighted mean (as opposed to sum).
            normalizer = torch.sum(attn, axis=-1, keepdim=True) + self.epsilon
            attn_n = attn / normalizer
        else:
            attn_n = attn

        if self.attn_weights_only:
            return attn_n

        # Aggregate values using a weighted sum with weights provided by `attn`
        updates = torch.einsum("bhqk,bkhd->bqhd", attn_n, value)

        return updates, attn


def read_path(
    root: Any, path: Optional[str] = None, elements: Optional[List[str]] = None, error: bool = True
):
    if path is not None and elements is None:
        elements = path.split(".")
    elif path is None and elements is None:
        raise ValueError("`elements` and `path` can not both be `None`")

    current = root
    for elem in elements:
        if isinstance(current, Mapping):
            next = current.get(elem)
            if next is None:
                if not error:
                    return None
                if path is None:
                    path = ".".join(elements)
                raise ValueError(
                    f"Can not use element {elem} of path `{path}` to access into "
                    f"dictionary. Available options are {', '.join(list(current))}"
                )
        elif isinstance(current, Sequence):
            try:
                index = int(elem)
            except ValueError:
                if not error:
                    return None
                if path is None:
                    path = ".".join(elements)
                raise ValueError(
                    f"Element {elem} of path `{path}` can not be converted to index into sequence."
                ) from None

            try:
                next = current[index]
            except IndexError:
                if not error:
                    return None
                if path is None:
                    path = ".".join(elements)
                raise ValueError(
                    f"Can not use element {elem} of path `{path}` to access into "
                    f"sequence of length {len(current)}"
                ) from None
        elif hasattr(current, elem):
            next = getattr(current, elem)
        else:
            if not error:
                return None
            if path is None:
                path = ".".join(elements)
            raise ValueError(
                f"Can not handle datatype {type(current)} at element " f"{elem} of path `{path}`"
            )

        current = next

    return current


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")


class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str = "processor.state",
        temperature: float = 0.1,
        batch_contrast: bool = True,
        **kwargs,
    ):
        # Important:
        # The parent Loss class normally flattens video patch dimensions.
        # We do NOT want that. We need [B, T, K, D].
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)

        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            video_inputs=True,
            patch_inputs=False,
            keep_input_dim=True,
            **kwargs,
        )

        self.temperature = float(temperature)
        self.batch_contrast = bool(batch_contrast)
        self.criterion = nn.CrossEntropyLoss()

    def get_target(self, inputs, outputs):
        # Target is unused; return a dummy tensor so the existing loss pipeline works.
        # This avoids wasting memory by copying large tensors as targets.
        pred = read_path(outputs, elements=self.pred_path)
        return torch.empty((), device=pred.device, dtype=pred.dtype)

    def forward(self, slots: torch.Tensor, target: torch.Tensor = None):
        """
        slots: [B, T, K, D]
        B = batch size
        T = video length
        K = number of slots
        D = slot dimension
        """
        if slots.ndim != 4:
            raise ValueError(
                f"Slot contrast expects [B, T, K, D], but got {tuple(slots.shape)}"
            )

        B, T, K, D = slots.shape
        if T < 2:
            return slots.new_tensor(0.0)

        slots = F.normalize(slots, p=2.0, dim=-1)

        s_prev = slots[:, :-1]  # [B, T-1, K, D]
        s_next = slots[:, 1:]   # [B, T-1, K, D]

        if self.batch_contrast:
            # For each time step, contrast each slot against all slots
            # from all videos in the batch at the next frame.
            #
            # q: [T-1, B*K, D]
            # k: [T-1, B*K, D]
            q = s_prev.permute(1, 0, 2, 3).reshape(T - 1, B * K, D)
            k = s_next.permute(1, 0, 2, 3).reshape(T - 1, B * K, D)

            # logits[t, i, j] = similarity between previous slot i and next slot j
            logits = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
            logits = logits.reshape((T - 1) * B * K, B * K)

            # Positive pair: same video, same slot index, next frame.
            labels = torch.arange(B * K, device=slots.device)
            labels = labels.unsqueeze(0).expand(T - 1, B * K).reshape(-1)

        else:
            # Only contrast slots within the same video.
            # logits: [B, T-1, K, K]
            logits = torch.matmul(s_prev, s_next.transpose(-2, -1)) / self.temperature
            logits = logits.reshape(B * (T - 1) * K, K)

            labels = torch.arange(K, device=slots.device)
            labels = labels.view(1, 1, K).expand(B, T - 1, K).reshape(-1)

        return self.criterion(logits, labels)

class SlotToTopPosterior(nn.Module):
    """
    Maps slot posterior parameters directly to VDVAE top posterior map.

    Inputs:
        slot_mu:       [B, S, D]
        slot_logsigma: [B, S, D]
        cond_top:      [B, Ccond, H, W] = concat(h_t, z_tilde_t)
    Outputs:
        qm:            [B, C, H, W]
        qv:            [B, C, H, W]  # IMPORTANT: logsigma, not logvar
        z_top_map:     [B, C, H, W]
        comp_mu:       [B, S, C, H, W]
        comp_logsigma: [B, S, C, H, W]
        masks:         [B, S, 1, H, W]
    """

    def __init__(self, slot_dim:int, z_channels:int, top_h:int, top_w:int, cond_channels: int, cond_width: int, use_3x3: bool = True):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.z_channels = int(z_channels)
        self.top_h = int(top_h)
        self.top_w = int(top_w)
        self.cond_channels = int(cond_channels)

        # the inputt would be slot and positional encoding and hidden state of RNN + z_tilde_t
        self.net = Block(
                2 * slot_dim + 4 + cond_channels,
                cond_width,
                2 * self.z_channels,
                residual=False,
                use_3x3=use_3x3,
            )

    def _coord_grid(self, B, S, device, dtype):
        y = torch.linspace(-1.0, 1.0, self.top_h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, self.top_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        grid = torch.stack([xx, yy, 1.0 - xx, 1.0 - yy], dim=0)
        return grid[None, None].expand(B, S, 4, self.top_h, self.top_w)

    def forward(self, slot_mu, slot_logsigma, cond_top):
        B, S, D = slot_mu.shape
        assert D == self.slot_dim, (D, self.slot_dim)
        slot_logsigma = slot_logsigma.clamp(-7.0, 3.0)
        slot_params = torch.cat([slot_mu, slot_logsigma], dim=-1)

        slot_grid = slot_params[:, :, :, None, None].expand(
            B, S, 2 * D, self.top_h, self.top_w
        )

        pos = self._coord_grid(B, S, slot_mu.device, slot_mu.dtype)

        cond = cond_top[:, None].expand(B, S, self.cond_channels, self.top_h, self.top_w)

        inp = torch.cat([slot_grid, pos, cond], dim=2)
        inp = inp.reshape(B * S, 2 * D + 4 + self.cond_channels, self.top_h, self.top_w)

        out = self.net(inp)
        out = out.view(B, S, 2 * self.z_channels, self.top_h, self.top_w)

        comp_mu = out[:, :, :self.z_channels]
        comp_logsigma = out[:, :, self.z_channels:2 * self.z_channels]

        comp_logsigma = comp_logsigma.clamp(-7.0, 3.0)

        # Per-slot latent maps. These are what should go into the decoder.
        z_slot_maps = comp_mu + torch.exp(comp_logsigma) * torch.randn_like(comp_mu)

        # Aggregate only for top-level KL / temporal state, not for image decoding.
        qm = comp_mu.mean(dim=1)

        comp_var = torch.exp(2.0 * comp_logsigma)
        second_moment = (comp_var + comp_mu.pow(2)).mean(dim=1)
        q_var = (second_moment - qm.pow(2)).clamp_min(1e-8)
        qv = 0.5 * torch.log(q_var)

        # This mixed z is only for temporal/RNN bookkeeping if you still need it.
        z_top_map = qm + torch.exp(qv) * torch.randn_like(qm)

        return qm, qv, z_top_map, comp_mu, comp_logsigma, z_slot_maps


class TopSlotPosterior(nn.Module):
    def __init__(
        self,
        in_channels,
        z_channels,
        top_h,
        top_w,
        cond_channels,
        cond_width,
        num_slots,
        slot_dim,
        n_iters=3,
        logsigma_clamp=(-5.0, 3.0),
        use_checkpoint=True,
    ):
        super().__init__()
        self.top_h = int(top_h)
        self.top_w = int(top_w)
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.in_channels = int(in_channels)
        self.logsigma_clamp = logsigma_clamp
        self.use_checkpoint = bool(use_checkpoint)
        # Position embedding must now match encoder feature channels.
        self.pre_pos = SoftPositionEmbed(
            hidden_size=in_channels,
            resolution=(top_h, top_w),
        )
        self.pre_norm = nn.LayerNorm(in_channels)

        self.slots_init_mu = nn.Embedding(num_slots, slot_dim)
        nn.init.orthogonal_(self.slots_init_mu.weight)

        self.slots_init_logsigma = nn.Embedding(num_slots, slot_dim)
        nn.init.xavier_uniform_(self.slots_init_logsigma.weight)

        # Slot Attention receives encoder feature dim,
        # but slots themselves still live in slot_dim.
        slot_weight_init = {
            "linear_w": "xavier_uniform",
            "linear_b": "zeros",
        }

        self.slot_attention = SlotAttention(
            input_size=in_channels,
            slot_size=slot_dim,
            qkv_size=slot_dim,
            num_iterations=n_iters,
            mlp_size=slot_dim,
            num_heads=1,
            weight_init=slot_weight_init,
        )

        self.to_mu = nn.Linear(slot_dim, slot_dim)
        self.to_logsigma = nn.Linear(slot_dim, slot_dim)

        self.slot_to_top = SlotToTopPosterior(
            slot_dim=slot_dim,
            z_channels=z_channels,
            top_h=top_h,
            top_w=top_w,
            cond_channels=cond_channels,
            cond_width=cond_width,
            use_3x3=True,
        )

    def _maybe_ckpt(self, fn, *args):
        if (
            self.training
            and self.use_checkpoint
            and any(torch.is_tensor(a) and a.requires_grad for a in args)
        ):
            return ckpt(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, acts_top, cond_top):
        B, C, H, W = acts_top.shape
        assert H == self.top_h and W == self.top_w, (H, W, self.top_h, self.top_w)
        assert C == self.in_channels, (C, self.in_channels)

        # Directly use encoder features.
        feat = self.pre_pos(acts_top)

        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        tokens = self.pre_norm(tokens)
        
        init_mu = self.slots_init_mu(torch.arange(self.num_slots, device=acts_top.device)).unsqueeze(0).expand(B, -1, -1).to(dtype=tokens.dtype)
        init_logsigma = self.slots_init_logsigma(torch.arange(self.num_slots, device=acts_top.device)).unsqueeze(0).expand(B, -1, -1).to(dtype=tokens.dtype).clamp(*self.logsigma_clamp)
        init_slots = init_mu + torch.exp(init_logsigma) * torch.randn_like(init_mu)

        def _slot_attention_fn(init_slots_, tokens_):
            out = self.slot_attention(init_slots_, tokens_)
            return out["slots"], out["masks"]

        slots, slot_attn = self._maybe_ckpt(
            _slot_attention_fn,
            init_slots,
            tokens,
        )
        slot_mu = self.to_mu(slots)
        slot_logsigma = self.to_logsigma(slots).clamp(*self.logsigma_clamp)

        def _slot_to_top_fn(slot_mu_, slot_logsigma_, cond_top_):
            return self.slot_to_top(slot_mu_, slot_logsigma_, cond_top_)

        qm, qv, z_top_map, comp_mu, comp_logsigma, z_slot_maps = self._maybe_ckpt(
            _slot_to_top_fn,
            slot_mu,
            slot_logsigma,
            cond_top,
        )
        return {
            "slot_mu": slot_mu,
            "slot_logsigma": slot_logsigma,

            "top_q_mean_map": qm,
            "top_q_logsigma_map": qv,
            "z_top_map": z_top_map,

            "slot_attn": slot_attn,
            "slot_comp_maps": comp_mu,
            "slot_comp_logsigma": comp_logsigma,
            "z_slot_maps": z_slot_maps,
        }


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    logsigma1 = logsigma1.clamp(-7.0, 3.0)
    logsigma2 = logsigma2.clamp(-7.0, 3.0)
    kl = -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
    return kl.sum(dim=(1, 2, 3))

@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    logsigma = logsigma.clamp(-7.0, 3.0)
    return torch.exp(logsigma) * eps + mu


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, mask_logits, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    B = xs[0]
    N = ls[0]
    S = N // B
    H, W = xs[1], xs[2]
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    # [B,H,W,3] -> [B,S,H,W,3] -> [B*S,H,W,3]
    x = x[:, None].expand(B, S, H, W, 3).reshape(N, H, W, 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], [N, H, W, 3] + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, [N, H, W, 3] + [1]) + torch.zeros([N, H, W, 3] + [nr_mix]).to(device=x.device, dtype=x.dtype)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [N, H, W, 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [N, H, W, 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [N, H, W, 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, dim=-1)
    denom = float(H * W * 3)
    logp_slots = mixture_probs.view(B, S, H, W)
    mask_logits = mask_logits.reshape(B, S, H, W)
    log_masks = F.log_softmax(mask_logits, dim=1)  # [B,S,H,W]

    # log p(x_hw) = logsumexp_s(log m_s_hw + log p_s(x_hw))
    logp_pixel = torch.logsumexp(log_masks + logp_slots, dim=1)  # [B,H,W]

    return -1. * logp_pixel.sum(dim=[1, 2]) / denom


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)

def mean_from_discretized_mix_logistic(l: torch.Tensor, nr_mix: int) -> torch.Tensor:
    """
    Differentiable mean reconstruction from DMOL parameters.

    Args:
        l: [B, H, W, 10*nr_mix] - output of DmolNet.forward()
        nr_mix: number of mixture components

    Returns:
        [B, H, W, 3] in [-1, 1], differentiable w.r.t. l
    """
    B, H, W, _ = l.shape

    # Mixture weights
    logit_probs = l[:, :, :, :nr_mix]
    probs = F.softmax(logit_probs, dim=-1)  # [B,H,W,M]

    # Reshape remaining params: [B,H,W,9M] -> [B,H,W,3,3M]
    params = l[:, :, :, nr_mix:].reshape(B, H, W, 3, 3 * nr_mix)

    # Extract per-channel params
    means = params[:, :, :, :, :nr_mix]                          # [B,H,W,3,M]
    coeffs = torch.tanh(params[:, :, :, :, 2*nr_mix:3*nr_mix])   # [B,H,W,3,M]

    # Autoregressive conditional expectations per mixture
    # Note: coeffs[:,…,i,:] is the i-th AR coefficient, not channel i's coeffs
    mu_r = means[:, :, :, 0, :]  # [B,H,W,M]
    mu_g = means[:, :, :, 1, :]
    mu_b = means[:, :, :, 2, :]

    c_rg = coeffs[:, :, :, 0, :]  # R → G coefficient
    c_rb = coeffs[:, :, :, 1, :]  # R → B coefficient
    c_gb = coeffs[:, :, :, 2, :]  # G → B coefficient

    # E[x|k] for each mixture component
    e_r_k = mu_r
    e_g_k = mu_g + c_rg * e_r_k
    e_b_k = mu_b + c_rb * e_r_k + c_gb * e_g_k

    # Marginal expectation: E[x] = Σ_k π_k E[x|k]
    e_r = (probs * e_r_k).sum(dim=-1)
    e_g = (probs * e_g_k).sum(dim=-1)
    e_b = (probs * e_b_k).sum(dim=-1)

    out = torch.stack([e_r, e_g, e_b], dim=-1)
    return out.clamp(-1., 1.)

class HModule(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.build()

"""
class DmolNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_conv(H.width, H.num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), low_bit=self.H.dataset in ['ffhq_256'])

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        xhat = (im + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat
"""
class DmolNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_conv(H.width, H.num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x):
        
        mask_logits = px_z[:, self.width:self.width + 1]

        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), mask_logits=mask_logits, low_bit=self.H.dataset in ['ffhq_256'])

    def forward(self, px_z):
        xhat = self.out_conv(px_z[:, :self.width])
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        N, H, W, _ = im.shape
        S = self.H.top_num_slots
        B = N // S
        mask_logits = px_z[:, self.width:self.width + 1]  # [B*S,1,H,W]
        mask_logits = mask_logits.view(B, S, 1, H, W)
        masks = F.softmax(mask_logits, dim=1)             # [B,S,1,H,W]

        im = im.view(B, S, H, W, 3)
        im = torch.sum(im * masks.permute(0, 1, 3, 4, 2), dim=1)
        xhat = (im + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat


class Block(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)        
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out
