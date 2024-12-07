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
from torch.nn import functional as F

from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.beatmapgen_modules import OutputLMModel
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
    def __init__(self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider, beatmap_pattern_provider: CodebooksPatternProvider,
                 fuser: ConditionFuser, token_id_size: int, position_size: int, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8, num_layers: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False, difficulty_num: int = 5, 
                 transfer_dim: int = 64, transfer_num_heads: int = 4, transfer_num_layers: int = 1,
                 use_mask: bool = False, lora_kwargs: dict = {}, transfer_lm_kwargs: dict = {}, transfer_lr: tp.Optional[float] = None, 
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
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        self.difficulty_num = difficulty_num
        # beatmap token id
        self.token_id_size = token_id_size + 1 # token_id_size is no_note_id, token_id_size + 1 is padding_id, token_id_size + 2 is actual size
        
        self.position_size = position_size
        self.use_mask = use_mask
        self.difficulty_emb = ScaledEmbedding(self.difficulty_num, transfer_dim, lr=transfer_lr)
        self.beatmap_emb = nn.ModuleList([ScaledEmbedding(self.token_id_size, transfer_dim, lr=transfer_lr) for _ in range(self.position_size)])
        self.linear_out = nn.ModuleList([nn.Linear(transfer_dim, self.token_id_size, bias=bias_proj) for _ in range(self.position_size)])
        self.transfer_dim = transfer_dim
        self.local_cross_attention = transfer_lm_kwargs['local_cross_attention']
        self.beatmap_pattern_provider = beatmap_pattern_provider
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            lora_kwargs = lora_kwargs, d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim), num_layers = num_layers,
            norm=norm, norm_first=norm_first, **kwargs)
        self.out_norm: tp.Optional[nn.Module] = None
        self.out_norm2: tp.Optional[nn.Module] = None
        if norm_first: 
            self.out_norm = create_norm_fn(norm, dim)
            self.out_norm2 = create_norm_fn(norm, transfer_dim)
        self.transfer_lm = StreamingTransformer(
            d_model=transfer_dim, num_heads=transfer_num_heads, dim_feedforward=int(hidden_scale * transfer_dim), num_layers = transfer_num_layers,
            norm=norm, norm_first=norm_first, position_size = position_size, transfer_lm_kwargs = transfer_lm_kwargs, lr = transfer_lr, **kwargs)
        self.linear_transfer = nn.Linear(dim, self.transfer_dim, bias=bias_proj)
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        if self.use_mask:
            self.mask_token_embedding = nn.Parameter(torch.empty((self.dim,)))
            nn.init.uniform_(self.mask_token_embedding, -0.1, 0.1)

        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

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

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        init_layer(self.difficulty_emb, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        
        for emb_layer in self.beatmap_emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        transformer_to_initialize = [self.transformer.layers, self.transfer_lm.layers]
        for module in transformer_to_initialize:
            for layer_idx, tr_layer in enumerate(module):
                depth = None
                if depthwise_init == 'current':
                    depth = layer_idx + 1
                elif depthwise_init == 'global':
                    depth = len(self.transformer.layers)
                init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
                tr_layer.apply(init_fn)

        init_layer(self.linear_transfer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        
        for linear in self.linear_out:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        
    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(self, codes: torch.Tensor,
                note_code_maps: list,
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
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id
        )
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)]) # batch, sequence, dim
        if self.use_mask:
            mask_positions = [[pos + 4 for pos in positions] for positions in note_code_maps]
            for i, positions in enumerate(mask_positions):
                input_[i, positions, :] += self.mask_token_embedding
        out = self.transformer(input_, cross_attention_src=None,
                               src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None))
        if self.out_norm:
            out = self.out_norm(out)
        out = out[:,4:,:]

        return out  #[B, S, dim]

    def compute_predictions(
            self, codes: torch.Tensor,
            beatmap: torch.Tensor,
            difficulty: torch.Tensor,
            note_code_maps: list,
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
        # use musicgen output as cross attention source for transfer LM
        model = self if self._fsdp is None else self._fsdp
        out = model(codes, note_code_maps, stage=stage)
        
        logit_list = []
        for one_out, note_code_map, one_beatmap, one_difficulty in zip(out, note_code_maps, beatmap, difficulty):
            # use note represenetation
            cross_attention_input = self.linear_transfer(one_out[note_code_map]).unsqueeze(0)
            codes = one_beatmap[note_code_map].unsqueeze(0).permute(0, 2, 1) 
            B, K, T = codes.shape
            codes = codes.contiguous()
            pattern = self.beatmap_pattern_provider.get_pattern(T)
            sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
                codes, self.token_id_size, keep_only_valid_steps=keep_only_valid_steps,
            )
            sequence_codes = sequence_codes[:, :, 1:-1]
            logits = self.transfer_lm_forward(sequence=sequence_codes,cross_attention_input = cross_attention_input, difficulty= one_difficulty.unsqueeze(0).unsqueeze(0))
            logits = F.pad(logits, (0, 0, 0, 1))
            logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
            logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
                logits, float('nan'), keep_only_valid_steps=keep_only_valid_steps
            )
            logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
            logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
            logit_list.append(LMOutput(logits, logits_mask))
        return logit_list
    
    def transfer_lm_forward(self, 
                cross_attention_input: torch.Tensor,
                sequence: tp.Optional[torch.Tensor] = None,
                difficulty: tp.Optional[torch.Tensor] = None,
                stage: int = -1) -> torch.Tensor:
        if sequence is not None:
            B, K, S = sequence.shape
            assert K == self.position_size, "Sequence shape must match the specified number of codebooks"
            input_ = sum([self.beatmap_emb[k](sequence[:, k]) for k in range(K)])
            if difficulty is not None:
                input_ = torch.cat((self.difficulty_emb(difficulty), input_), dim=1)
        else:
            assert difficulty is not None
            input_ = self.difficulty_emb(difficulty)
        out = self.transfer_lm(input_, cross_attention_src=cross_attention_input,
                            src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None)) # [B, S*P, dim] / [B, S, dim]
        if self.out_norm2:
            out = self.out_norm2(out) 
        
        logits = torch.stack([self.linear_out[p](out) for p in range(self.position_size)], dim=1)
        return logits

    def _sample_next_token(self,
                           sequence: torch.Tensor,
                           cross_attention_input: torch.Tensor,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None,
                           two_step_cfg: tp.Optional[bool] = None,
                           difficulty: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
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
        logits = self.transfer_lm_forward(sequence = sequence, cross_attention_input = cross_attention_input, difficulty = difficulty) # [1, 1, card]
        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]
        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token


    @torch.no_grad()
    def generate(
            self, codes: torch.Tensor,
            difficulty: torch.Tensor,
            note_code_maps: list,
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

        # use musicgen output as cross attention source for transfer LM
        model = self if self._fsdp is None else self._fsdp
        out = model(codes, note_code_maps, stage=stage)
        
        out_codes_list = []
        for one_out, one_difficulty, note_code_map in zip(out, difficulty, note_code_maps):
             # use note represenetation
            cross_attention_input = self.linear_transfer(one_out[note_code_map])
            
            max_gen_len = len(note_code_map)
            pattern = self.beatmap_pattern_provider.get_pattern(max_gen_len)
            unknown_token = -1
            gen_codes = torch.full((1, self.position_size, max_gen_len), unknown_token, dtype=torch.long, device=device) # gencodes.shape [3, 4, 400] max_gen_len = 400
            gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.token_id_size) # gen_sequence.shape 如果没有prompt, [3,4,404] 3
            start_offset = 0
            start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset) # start_offset = 0, start_offset_sequence = 1
            assert start_offset_sequence is not None
            B = 1
            with self.streaming():
                unconditional_state = self.get_streaming_state()
                prev_offset = 0
                gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [(S+1) * P]
                for offset in range(start_offset_sequence, gen_sequence_len):
                    # get current sequence (note that the streaming API is providing the caching over previous offsets)
                    curr_sequence = gen_sequence[..., prev_offset:offset] # shape is [B, K, 1]
                    curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                    if check:
                        # check coherence between mask and sequence
                        assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.token_id_size)).all()
                        # should never happen as gen_sequence is filled progressively
                        assert not (curr_sequence == unknown_token).any()
                    # sample next token from the model, next token and curr_sequence shape is [1]
                    if self.local_cross_attention:
                        index = offset - 1
                        cross_attention_src = cross_attention_input[index].unsqueeze(0).unsqueeze(0)
                    else:
                        cross_attention_src = cross_attention_input.unsqueeze(0)
                    if offset == 1:
                        one_difficulty = one_difficulty.unsqueeze(0).unsqueeze(0)
                        curr_sequence = None
                    else:
                        one_difficulty = None
                    next_token = self._sample_next_token(
                        curr_sequence, cross_attention_src, unconditional_state, use_sampling, temp, top_k, top_p,
                        cfg_coef=cfg_coef, two_step_cfg=two_step_cfg, difficulty=one_difficulty)
                    # ensure the tokens that should be masked are properly set to special_token_id
                    # as the model never output special_token_id
                    valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                    next_token[~valid_mask] = self.token_id_size
                    # ensure we don't overwrite prompt tokens, we only write over unknown tokens
                    # (then mask tokens should be left as is as well, which is correct)
                    gen_sequence[..., offset:offset+1] = torch.where(
                        gen_sequence[..., offset:offset+1] == unknown_token,
                        next_token, gen_sequence[..., offset:offset+1]
                    )
                    prev_offset = offset
                    if callback is not None:
                        callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
            unconditional_state.clear()
            # ensure sequence has been entirely filled
            assert not (gen_sequence == unknown_token).any()
            # ensure gen_sequence pattern and mask are matching
            # which means the gen_sequence is valid according to the pattern
            assert (
                gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.token_id_size)
            ).all()
            # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
            out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

            # sanity checks over the returned codes and corresponding masks
            assert (out_codes[..., :max_gen_len] != unknown_token).all()
            assert (out_mask[..., :max_gen_len] == 1).all()

            out_start_offset = start_offset if remove_prompts else 0
            out_codes = out_codes[..., out_start_offset:max_gen_len]

            # ensure the returned codes are all valid
            assert (out_codes >= 0).all() and (out_codes <= self.token_id_size).all()
            out_codes.permute(0, 2, 1)
            out_codes_list.append(out_codes)
        return out_codes_list
        
