# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn
from xformers.ops import fmha

from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn
from audiocraft.modules.transformer import set_efficient_attention_backend

logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).
    """
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class BeatmapLMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (MusicConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        n_q (int): Number of parallel streams to model.
        card (int): Cardinality, vocabulary size.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        emb_lr (float, optional): Embedding-specific learning rate.
        bias_proj (bool): Use bias for output projections.
        weight_init (str, optional): Method for weight initialization.
        depthwise_init (str, optional): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_dropout (float): Classifier-free guidance dropout.
        cfg_coef (float): Classifier-free guidance coefficient.
        attribute_dropout (dict): Attribute dropout probabilities.
        two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """
    def __init__(self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider,
                 fuser: ConditionFuser, token_id_size: int, position_size: int, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8, num_layers: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False, difficulty_num: int = 5, 
                 transfer_dim: int = 64, transfer_num_heads: int = 4, transfer_num_layers: int = 1,
                 use_mask: bool = False, lora_kwargs: dict = {}, blockwise_attention_kwargs: dict = {}, transfer_lr: tp.Optional[float] = None,
                 transfer_efficient_backend: str = 'torch', representation_dim: int = 128, representation: str = "spectrogram", segment_duration: int = 512, 
                 use_receptive_field: bool = False, ca_window_size: int = 3,
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 1
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        # self.two_step_cfg = two_step_cfg
        # self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        self.difficulty_num = difficulty_num
        self.block_self_attention = blockwise_attention_kwargs['block_self_attention']
        # beatmap token id
        self.token_id_size = token_id_size + 1
        self.position_size = position_size
        self.use_mask = use_mask
        self.transfer_efficient_backend = transfer_efficient_backend
        self.representation = representation
        self.segment_duration = segment_duration
        self.use_receptive_field = use_receptive_field
        self.ca_window_size = ca_window_size
        if self.block_self_attention:
            self.difficulty_emb = ScaledEmbedding(self.difficulty_num, transfer_dim * position_size, lr=transfer_lr)
            self.beatmap_emb = ScaledEmbedding(self.token_id_size, transfer_dim, lr=transfer_lr)
            self.linear_out = nn.Linear(transfer_dim, self.token_id_size, bias=bias_proj)
        else:
            self.difficulty_emb = ScaledEmbedding(self.difficulty_num, transfer_dim, lr=transfer_lr)
            self.beatmap_emb = nn.ModuleList([ScaledEmbedding(self.token_id_size, transfer_dim, lr=transfer_lr) for _ in range(self.position_size)])
            self.linear_out = nn.ModuleList([nn.Linear(transfer_dim, self.token_id_size, bias=bias_proj) for _ in range(self.position_size)])
        self.transfer_dim = transfer_dim
        self.local_cross_attention = blockwise_attention_kwargs['local_cross_attention']
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        # self.transformer = StreamingTransformer(
        #     lora_kwargs = lora_kwargs, d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim), num_layers = num_layers,
        #     norm=norm, norm_first=norm_first, **kwargs)
        # self.out_norm: tp.Optional[nn.Module] = None
        self.out_norm2: tp.Optional[nn.Module] = None
        if norm_first: 
            # self.out_norm = create_norm_fn(norm, dim)
            self.out_norm2 = create_norm_fn(norm, transfer_dim)
        self.transfer_lm = StreamingTransformer(
            d_model=transfer_dim, num_heads=transfer_num_heads, dim_feedforward=int(hidden_scale * transfer_dim), num_layers = transfer_num_layers,
            norm=norm, norm_first=norm_first, position_size = position_size, blockwise_attention_kwargs = blockwise_attention_kwargs, block_self_attention = self.block_self_attention, lr = transfer_lr, **kwargs)
        representation_dim = self.dim if self.representation == "musicgen" else representation_dim
        self.linear_transfer = nn.Linear(representation_dim, self.transfer_dim, bias=bias_proj)
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        if self.use_mask:
            self.mask_token_embedding = nn.Parameter(torch.empty((self.dim,)))
            nn.init.uniform_(self.mask_token_embedding, -0.1, 0.1)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

        self.attn_mask_for_sa = self.get_mask_transfer_lm(causal = True)
        self.attn_mask_for_ca = self.get_mask_transfer_lm(causal = False)

    def get_mask_transfer_lm(self, causal):
        # N = self.position_size
        # query_length = query.shape[1]
        # if self.block_self_attention:
        #     assert query_length % self.position_size == 0
        # beatmapgen inference case
        # if self._is_streaming:
        #     attn_mask = None
        # beatmapgen sa training case
        if causal:
            if self.block_self_attention:
                # custom_attn_mask = True
                # if not self.local_self_attention:
                query_len = self.segment_duration * self.position_size
                key_len = self.segment_duration * self.position_size
                d = torch.arange(self.segment_duration, device = 'cuda').repeat_interleave(self.position_size)
                dT = torch.arange(self.segment_duration, device = 'cuda').repeat_interleave(self.position_size)
                mask_op = ">="
                # else:
                    # add full block attention along diagonal line and tringular mask within block size
                    # raise RuntimeError("Not supported at the moment")
                    # attn_mask = torch.zeros(1, 1, query_length, query_length, dtype=torch.bool)
                    # for i in range(0, query_length, N):
                    #     index = i - N * self.sa_window_size
                    #     if index < 0:
                    #         index = 0
                    #     attn_mask[..., i:i+N, index:i+N] = True
            # else:
            #     if not self.local_self_attention:
                    # raise RuntimeError("Not supported at the moment")
                    # custom_attn_mask = False
                    # is_causal = True
                    # if _efficient_attention_backend == 'xformers':
                    #     attn_mask = LowerTriangularMask()
                # else:
                #     raise RuntimeError("Not supported at the moment")
                    # custom_attn_mask = True
                    # attn_mask = torch.zeros(1, 1, query_length, query_length, dtype=torch.bool)
                    # for i in range(0, query_length, 1):
                    #     index = i - self.sa_window_size
                    #     if index < 0:
                    #         index = 0
                    #     attn_mask[..., i, index:i+1] = True
        # beatmapgen ca training case
        else:
            if not self.local_cross_attention: # full cross attention case
                raise RuntimeError("Not supported at the moment")
                # attn_mask = None
            else:
                
                # if self.block_self_attention:
                #     assert (query_length // self.position_size) == key.shape[1]
                #     assert (query_length // self.position_size) == value.shape[1]
                # else:
                #     assert query_length == key.shape[1]
                #     assert query_length == value.shape[1]
                # cross_src_length = key.shape[1]
                # custom_attn_mask = True
                # if self.block_cross_attention or not self.block_self_attention:
                #     raise RuntimeError("Not supported at the moment")
                #     # attn_mask = torch.eye(cross_src_length, cross_src_length, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
                # else:
                query_len = self.segment_duration * self.position_size
                d = torch.arange(self.segment_duration, device = 'cuda').repeat_interleave(self.position_size)
                if self.use_receptive_field:
                    windows = self.ca_window_size*1 + 1
                    dT = torch.arange(self.segment_duration, device = 'cuda').repeat_interleave(windows)
                    key_len = self.segment_duration * windows
                else:
                    dT = torch.arange(self.segment_duration, device = 'cuda')
                    key_len = self.segment_duration
                mask_op = "=="
        d = d.view(1, -1, 1)
        dT = dT.view(1, -1, 1).transpose(1, 2)
        if mask_op == ">=":
            mask = d >= dT
        elif mask_op == "==":
            mask = d == dT
        zero_tensor = torch.full((1,), 0.0, device = 'cuda', dtype = torch.float16)
        neg_inf_tensor = torch.full((1,), float('-inf'), device = 'cuda', dtype = torch.float16)
        attn_mask = torch.where(mask, zero_tensor, neg_inf_tensor).reshape(1, 1, query_len, key_len)
        return attn_mask

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        # for emb_layer in self.emb:
        #     init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        init_layer(self.difficulty_emb, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        if self.block_self_attention:
            init_layer(self.beatmap_emb, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        else:
            for emb_layer in self.beatmap_emb:
                init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        if self.block_self_attention:
            init_layer(self.transfer_lm.local_pos_embedding, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        # transformer_to_initialize = [self.transformer.layers, self.transfer_lm.layers]
        transformer_to_initialize = [self.transfer_lm.layers]
        for module in transformer_to_initialize:
            for layer_idx, tr_layer in enumerate(module):
                depth = None
                if depthwise_init == 'current':
                    depth = layer_idx + 1
                elif depthwise_init == 'global':
                    depth = len(module)
                init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
                tr_layer.apply(init_fn)

        init_layer(self.linear_transfer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        if self.block_self_attention:
            init_layer(self.linear_out, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        else:
            for linear in self.linear_out:
                init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
            
    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(self, sequence: torch.Tensor,
                src_mask: tp.Optional[torch.Tensor] = None,
                # note_code_maps: list,
                # conditions: tp.List[ConditioningAttributes],
                # condition_tensors: tp.Optional[ConditionTensors] = None,
                stage: int = -1) -> torch.Tensor:
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, a tensor of beatmap of shape [B, S, P] with S the sequence steps and
        P the number of beatmap positions, difficulty with shape [B]
        return the logits with shape [B, S, P, card].
        
        Args:
            indices (torch.Tensor): Indices of the codes to model.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
            stage (int): The codebook level that is being predicted. Relevant for MAGNeT
                in which prediction is done in a codebook-by-codebook manner.
                Takes values in range(n_q), and ignored by default.
        Returns:
            torch.Tensor: Logits.
        """
        set_efficient_attention_backend("xformers")
        B, K, T = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)]) # batch, sequence, dim
        # if self.use_mask:
        #     mask_positions = [[pos + 4 for pos in positions] for positions in note_code_maps]
        #     for i, positions in enumerate(mask_positions):
        #         input_[i, positions, :] += self.mask_token_embedding
        out = self.transformer(input_, cross_attention_src=None,
                               src_mask = src_mask)
        if self.out_norm:
            out = self.out_norm(out)

        return out  #[B, S, dim]

    def compute_representation(self, codes: torch.Tensor) -> torch.Tensor:
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id
        )
        
        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        max_gen_len = sequence.shape[-1]
        gen_representation = model(sequence, None)
        # with self.streaming():
        #     gen_representation = model(sequence[:,:,:window_size], None)
        #     if max_gen_len > window_size:
        #         half_window_size = window_size // 2
        #         for offset in list(range(window_size, max_gen_len, half_window_size)):
        #             if offset + half_window_size > max_gen_len:
        #                 q_length = max_gen_len - offset
        #             else:
        #                 q_length = half_window_size
        #             state = self.get_streaming_state()
        #             for key, value in state.items():
        #                 if len(value.shape) == 4:
        #                     assert value.shape[1] == window_size
        #                     state[key] = value[:, half_window_size:]
        #             self.set_streaming_state(state)
        #             q_length_list = [q_length]*B
        #             q_seqinfo = fmha.attn_bias._SeqLenInfo.from_seqlens(q_length_list)
        #             batch_sizes = [1]*B
        #             k_length_list = [half_window_size+q_length]*B
        #             k_seqinfo = fmha.attn_bias._SeqLenInfo.from_seqlens(k_length_list)
        #             attn_mask = fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask(
        #                 q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, _batch_sizes=batch_sizes
        #             )
        #             gen_representation = torch.cat((gen_representation, model(sequence[:,:,offset:offset+q_length], attn_mask)), dim=1)
        # self.reset_streaming()
        assert gen_representation.shape[1] == max_gen_len
        return gen_representation

    def compute_predictions(
            self, codes: torch.Tensor,
            beatmap: torch.Tensor,
            difficulty: torch.Tensor,
            
            stage: int = -1,
            keep_only_valid_steps: bool = True):
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list of ConditioningAttributes): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning
                tensors, see `conditions`.
            stage (int): The codebook level that is being predicted. Relevant for MAGNeT
                in which prediction is done in a codebook-by-codebook manner.
                Takes values in range(n_q), and ignored by default.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token in that case.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """

        cross_attention_input = self.linear_transfer(codes)
        B = cross_attention_input.shape[0]
        if self.use_receptive_field:
            B, S, index, D = cross_attention_input.shape
            assert S == self.segment_duration, "segment duration should be the same as the input"
            cross_attention_input = cross_attention_input.view(cross_attention_input.shape[0], -1, self.transfer_dim)
        if self.block_self_attention:
            beatmap = beatmap[:,:-1].view(B, -1) # [B, (S-1)*P]
            input_ = self.beatmap_emb(beatmap) # [B, (S-1)*P, dim]
            input_ = torch.cat((self.difficulty_emb(difficulty).reshape(B, self.position_size, -1), input_), dim=1) #[B, S*P, dim]
        else:
            beatmap = beatmap[:, :-1] # [B, (S-1), P]
            input_ = sum([self.beatmap_emb[p](beatmap[:, :, p]) for p in range(self.position_size)]) # [B, (S-1), dim]
            input_ = torch.cat((self.difficulty_emb(difficulty).unsqueeze(1), input_), dim=1) #[B, S, dim]
        logits = self.transfer_lm_forward(input_ = input_, cross_attention_input = cross_attention_input, src_mask = self.attn_mask_for_sa, cross_src_mask = self.attn_mask_for_ca) # [1, S*P, card]
        return logits
    
    def transfer_lm_forward(self, input_: torch.Tensor, # [B, S*P, card]
                cross_attention_input: torch.Tensor,
                src_mask: torch.Tensor = None,
                cross_src_mask: torch.Tensor = None,
                stage: int = -1) -> torch.Tensor:
        set_efficient_attention_backend(self.transfer_efficient_backend)
        out = self.transfer_lm(input_, cross_attention_src=cross_attention_input,
                            src_mask=src_mask, cross_src_mask=cross_src_mask) # [B, S*P, dim] / [B, S, dim]
        if self.out_norm2:
            out = self.out_norm2(out) 
        if self.block_self_attention:
            logit = self.linear_out(out) # [B, S*P, card]
        else:
            logit = torch.stack([self.linear_out[p](out) for p in range(self.position_size)], dim=2)
            B, S, P, C = logit.shape
            logit = logit.view(B, -1, C)
        return logit

    def _sample_next_token(self,
                           input: torch.Tensor,
                           cross_attention_input: torch.Tensor,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None,
                           two_step_cfg: tp.Optional[bool] = None) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float, optional): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        logit = self.transfer_lm_forward(input_ = input, cross_attention_input = cross_attention_input) # [1, 1, card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logit / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logit, dim=-1, keepdim=True)

        return next_token


    @torch.no_grad()
    def generate(
            self, codes: torch.Tensor,
            difficulty: torch.Tensor,
            stage: int = -1,
            prompt: tp.Optional[torch.Tensor] = None,
            num_samples: tp.Optional[int] = None,
            use_sampling: bool = True,
            temp: float = 1.0,
            top_k: int = 250,
            top_p: float = 0.0,
            cfg_coef: tp.Optional[float] = None,
            two_step_cfg: tp.Optional[bool] = None,
            remove_prompts: bool = False,
            check: bool = False,
            callback: tp.Optional[tp.Callable[[int, int], None]] = None,
            **kwargs) -> torch.Tensor:
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device
        
        cross_attention_input = self.linear_transfer(codes)
        unknown_token = -1
        max_gen_len = (self.segment_duration + 1) * self.position_size
        B = cross_attention_input.shape[0]
        gen_sequence = torch.full((B, max_gen_len,), unknown_token, dtype=torch.long, device=device)
        gen_sequence[:, 0:self.position_size] = difficulty.unsqueeze(1)
        start_offset_sequence = self.position_size
        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, (S+1) * P]
            for offset in range(start_offset_sequence, gen_sequence_len, self.position_size):
                # get current sequence (note that the streaming API is providing the caching over previous offsets)
                curr_sequence = gen_sequence[:, prev_offset:offset]
                # sample next token from the model, next token and curr_sequence shape is [1]
                if offset == self.position_size:
                    if self.block_self_attention:
                        input = self.difficulty_emb(difficulty).reshape(B, self.position_size, -1)
                    else:
                        input = self.difficulty_emb(difficulty).unsqueeze(1)
                else:
                    if self.block_self_attention:
                        input = self.beatmap_emb(curr_sequence)
                    else:
                        input = sum([self.beatmap_emb[p](curr_sequence[:, p]) for p in range(self.position_size)]).unsqueeze(1)
                if self.local_cross_attention:
                    index = offset // self.position_size - 1
                    cross_attention_src = cross_attention_input[:, index]
                    if not self.use_receptive_field:
                        cross_attention_src = cross_attention_src.unsqueeze(1)
                else:
                    cross_attention_src = cross_attention_input
                next_token = self._sample_next_token(
                    input, cross_attention_src, unconditional_state, use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef, two_step_cfg=two_step_cfg)
                next_token = next_token.view(B, -1)
                # ensure we don't overwrite prompt tokens, we only write over unknown tokens
                # (then mask tokens should be left as is as well, which is correct)
                assert (gen_sequence[:, offset:offset+self.position_size] == unknown_token).any()
                gen_sequence[:, offset:offset+self.position_size] = torch.where(
                    gen_sequence[:, offset:offset+self.position_size] == unknown_token,
                    next_token, gen_sequence[:, offset:offset+self.position_size]
                )
                prev_offset = offset
                if callback is not None:
                    callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()
        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_code = gen_sequence[:, self.position_size:]
        # sanity checks over the returned codes and corresponding masks
        assert (out_code[:, :max_gen_len] != unknown_token).all()
        # ensure the returned codes are all valid
        assert (out_code >= 0).all() and (out_code <= self.token_id_size).all()
        out_code = out_code.view(B, self.segment_duration, self.position_size)

        return out_code
        
