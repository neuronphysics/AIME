import sys
from VRNN.perceiver.perceiver_blocks import *
from VRNN.perceiver.perceiver_helpers import *

# Perceiver model variants.
VARIANTS = {
    'Mini': {
        'num_groups': (16, 1, 16),
        'num_self_attends_per_block': (2, 1, 1),
        'z_index_dim': (128, 64, 128),
        'num_z_channels': (128, 256, 128),
        'num_cross_attend_heads': (1, 1, 1),
        'num_self_attend_heads': (4, 32, 4),
        'cross_attend_widening_factor': (1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4),
        'num_embedding_channels': 32,
    },
    '16': {
        'num_groups': (16, 4, 1, 1, 1, 4, 16),
        'num_self_attends_per_block': (2, 2, 18, 2, 1, 1, 1),
        'z_index_dim': (128, 256, 256, 64, 256, 256, 128),
        'num_z_channels': (128, 256, 512, 1024, 512, 256, 128),
        'num_cross_attend_heads': (1, 1, 1, 1, 1, 1, 1),
        'num_self_attend_heads': (4, 8, 16, 32, 16, 8, 4),
        'cross_attend_widening_factor': (1, 1, 1, 1, 1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4, 4, 4, 4, 4),
        'num_embedding_channels': 32,
    },
    '256': {
        'num_groups': (256, 64, 16, 4, 1, 1, 1, 4, 16, 64, 256),
        'num_self_attends_per_block': (1, 1, 2, 2, 18, 2, 1, 1, 1, 1, 1),
        'z_index_dim': (32, 64, 128, 256, 256, 64, 256, 256, 128, 64, 32),
        'num_z_channels': (64, 96, 128, 256, 512, 1024, 256, 128, 64, 32, 16),
        'num_cross_attend_heads': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'num_self_attend_heads': (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1),
        'cross_attend_widening_factor': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
        'num_embedding_channels': 16,
    },
    '16x3_lite': {
    'num_groups': (16, 1, 16),
    'num_self_attends_per_block': (1, 6, 1),
    'z_index_dim': (128, 128, 128),
    'num_z_channels': (128, 512, 128),       # narrower than current 1024 processor
    'num_cross_attend_heads': (1, 1, 1),
    'num_self_attend_heads': (4, 16, 4),
    'cross_attend_widening_factor': (1, 1, 1),
    'self_attend_widening_factor': (4, 4, 4),
    'num_embedding_channels': 64,            # <— key upgrade from 32
    },
    '16x3': {
        'num_groups': (16, 1, 16),
        'num_self_attends_per_block': (2, 18, 2),
        'z_index_dim': (128, 256, 128),
        'num_z_channels': (128, 1024, 128),
        'num_cross_attend_heads': (1, 1, 1),
        'num_self_attend_heads': (4, 32, 4),
        'cross_attend_widening_factor': (1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4),
        'num_embedding_channels': 32,
    },
    # Perceiver IO
    'io_mini': {
        'num_self_attends_per_block': 2,
        'z_index_dim': 128,
        'num_z_channels': 128,
        'num_cross_attend_heads': 1,
        'num_self_attend_heads': 2,
        'cross_attend_widening_factor': 1,
        'self_attend_widening_factor': 2,
        'num_embedding_channels': 128,
    },
    'io_c_50m': {
        'num_self_attends_per_block': 8,
        'z_index_dim': 1024,
        'num_z_channels': 512,
        'num_cross_attend_heads': 8,
        'num_self_attend_heads': 8,
        'cross_attend_widening_factor': 4,
        'self_attend_widening_factor': 4,
        'num_embedding_channels': 512,
    },
    'io_c_150m': {
        'num_self_attends_per_block': 12,
        'z_index_dim': 1024,
        'num_z_channels': 896,
        'num_cross_attend_heads': 16,
        'num_self_attend_heads': 16,
        'cross_attend_widening_factor': 4,
        'self_attend_widening_factor': 4,
        'num_embedding_channels': 896,
    },
}


def _check_and_get_processor_idx(num_groups: Sequence[int]) -> int:
    # The processor is the central block in a HiP.
    # [enc_1, ..., enc_N, processor, dec_1, ..., dec_N]
    processor_idx = len(num_groups) // 2
    # The processor block has 1 group: it is essentially a Perceiver IO.
    assert num_groups[processor_idx] == 1, 'The processor must use 1 group.'
    return processor_idx


class HiPClassBottleneck(nn.Module):
    def __init__(self,
                 input_data: Dict[str, torch.Tensor],
                 num_groups: List[int],
                 num_self_attends_per_block: List[int],
                 z_index_dim: List[int],
                 num_z_channels: List[int],
                 num_cross_attend_heads: List[int],
                 num_self_attend_heads: List[int],
                 cross_attend_widening_factor: List[int],
                 self_attend_widening_factor: List[int],
                 num_embedding_channels: int,
                 label_modalities: List[str],
                 num_position_encoding_channels: Optional[int] = None,
                 regroup_type: str = 'reshape',
                 activation_name: str = 'sq_relu',
                 processor_index_dim_train: Optional[int] = None,
                 processor_index_dim_eval: Optional[int] = None,
                 dropout_prob: float = 0.0,
                 drop_path_rate: float = 0.0,
                 task_type="classification",
                 unet_adapter_cfg: Optional[Dict[str, Any]] = None):
        super(HiPClassBottleneck, self).__init__()
        self.num_blocks = len(num_groups)
        assert self.num_blocks >= 3, 'At least 3 blocks are needed for U-Net residuals.'
        assert self.num_blocks % 2 == 1, 'HiP assumes an odd number of blocks.'
        self.regroup_type = regroup_type
        self.num_groups = num_groups
        self.task_type = task_type
        self.processor_block_idx = self.num_blocks // 2
        self.label_modalities = label_modalities
        self.class_label_inputs = {k: v for k, v in input_data.items() if k in self.label_modalities}

        self.grouper = ConstNumGrouper(num_groups=num_groups[0])
        self.embedder = Embedder(
            num_embedding_channels=num_embedding_channels,
            modalities=input_data,
            unet_adapter_cfg=unet_adapter_cfg,
        )
        self.position_encoder = PositionEncoder(
            modalities=input_data,
            num_position_encoding_channels=num_position_encoding_channels,
            embed_out_channel=num_embedding_channels
        )
        self.reconstruction_head = ReconstructionHead(num_groups=num_groups[-1], 
                                                      input_num_channel=num_z_channels[-1],
                                                      output_num_channels=num_embedding_channels,
                                                      output_index_dim_eval=int(input_data['image'].shape[1]),  # one query per pixel
                                                      drop_probs=dropout_prob)

        # Initialize Perceiver blocks for each level in the HiP hierarchy
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            block = PerceiverBlock(
                input_num_channel=num_embedding_channels if i == 0 else num_z_channels[i - 1],
                num_output_groups=num_groups[i], output_index_dim=z_index_dim[i],
                num_output_channels=num_z_channels[i],
                num_self_attend_layers=num_self_attends_per_block[i],
                num_self_attend_heads=num_self_attend_heads[i],
                self_attend_widening_factor=self_attend_widening_factor[i],
                num_cross_attend_heads=num_cross_attend_heads[i],
                cross_attend_widening_factor=cross_attend_widening_factor[i], regroup_inputs=(i > 0),
                regroup_type=regroup_type, activation_name=activation_name,
                output_index_dim_train=processor_index_dim_train if i == self.processor_block_idx else None,
                output_index_dim_eval=processor_index_dim_eval if i == self.processor_block_idx else None,
                dropout_prob=dropout_prob,
                drop_path_rate=drop_path_rate, 
                use_checkpoint=True)
            self.blocks.append(block)

    def forward(self, inputs: Dict[str, torch.Tensor], is_training: bool) -> Dict[str, torch.Tensor]:
        z_0 = self.embedder(inputs, un_embed=False)
        z, mae_query = self.position_encoder(z_0)
        z = self.grouper.group(z)
        mae_query = self.grouper.group(mae_query)

        hidden_z = []
        for i, block in enumerate(self.blocks):
            pre_attention_residual = hidden_z[self.num_blocks - i - 1] if i > self.processor_block_idx else None

            if i > 0:
                # Manually regroup the current latents to allow concatenation.
                # The grouper takes care of the initial regroup.
                z = regroup(
                    inputs=z,
                    num_output_groups=self.num_groups[i],
                    regroup_type=self.regroup_type)

            z = block(z, is_training=is_training, pre_attention_residual=pre_attention_residual)
            hidden_z.append(z)

        reconstruction_z_out = self.reconstruction_head(z, mae_query=mae_query, is_training=is_training)
        reconstruction_z_out = self.grouper.ungroup(reconstruction_z_out)
        reconstruction_output = self.embedder(reconstruction_z_out, un_embed=True)

        z_out = self.grouper.ungroup(z)
        output_keys = ModelOutputKeys
        return {
            output_keys.INPUT_RECONSTRUCTION: reconstruction_output,
            output_keys.LATENTS: z_out,
        }


def build_perceiver(input_data: Dict[str, torch.Tensor],
                    model_base_name: str,
                    model_variant_name: Optional[str],
                    model_kwargs: Optional[Dict[str, any]] = None,
                    searched_modules: Sequence[any] = [sys.modules[__name__]]):
    candidate = None
    for module in searched_modules:
        if hasattr(module, model_base_name):
            candidate = getattr(module, model_base_name)
            break

    assert candidate is not None, f'Failed to find class {model_base_name}.'

    if model_kwargs is None:
        model_kwargs = {}

    model_kwargs["input_data"] = input_data

    if model_variant_name is None:
        instance = candidate(**model_kwargs)
    else:
        assert model_variant_name in VARIANTS, f'VARIANTS does not contain {model_variant_name}.'
        instance = candidate(**model_kwargs, **VARIANTS[model_variant_name])

    return instance
