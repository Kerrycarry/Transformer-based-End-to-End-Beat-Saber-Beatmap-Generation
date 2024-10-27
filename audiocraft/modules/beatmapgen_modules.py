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

from ..utils import utils
from .streaming import StreamingModule, State
from .transformer import StreamingTransformer, create_norm_fn
from .conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from .codebooks_patterns import CodebooksPatternProvider
from .activations import get_activation_fn


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
    logits: torch.Tensor  # [B, K, T, token_id_size]
    mask: torch.Tensor  # [B, K, T]


class OutputLMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (MusicConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        token_id_size (int): token_id_size, vocabulary size.
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
    def __init__(self, input_dim: int, position_size: int, token_id_size: int = 1024, dim: int = 128, num_heads: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.position_size = position_size
        self.token_id_size = token_id_size
        embed_dim = self.token_id_size + 1
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_scale = hidden_scale
        self.emb = ScaledEmbedding(embed_dim, self.dim, lr=emb_lr)
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=self.dim, num_heads=self.num_heads, dim_feedforward=int(self.hidden_scale * self.dim),
            norm=norm, norm_first=norm_first, **kwargs)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, self.dim)
        self.linear = nn.Linear(self.dim, embed_dim, bias=bias_proj)
        self.input_linear = nn.Linear(self.input_dim, self.dim, bias=bias_proj)

        self._init_weights(weight_init, depthwise_init, zero_bias_init)
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

        
        init_layer(self.emb, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        
        init_layer(self.linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        init_layer(self.input_linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)


    @property
    def special_token_id(self) -> int:
        return self.token_id_size

    def forward(self, beatmap: torch.Tensor,
                condition: torch.Tensor,
                stage: int = -1) -> torch.Tensor:
        """Apply language model on beatmap and conditions.
        Given a tensor of beatmap of shape [B*S, P] with S the sequence steps and
        P the number of beatmap positions, condition of shape [B* S,1, dim], return the logits with shape [B* S, P, token_id_size].
        1 <=P <=13(12+1)
        """
        
        input_ = self.emb(beatmap)
        cross_attention_input = self.input_linear(condition)
        #input_ shape [B*S,P,dim], condition shape [B*S, 1, dim]
        out = self.transformer(input_, cross_attention_src=cross_attention_input,
                               src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None))
        if self.out_norm:
            out = self.out_norm(out)

        logits = self.linear(out) #[B*S, P, token_id_size]

        # remove the prefix from the model outputs
        # if len(self.fuser.fuse2cond['prepend']) > 0:
        #     logits = logits[:, :, -S:]
        
        return logits  #[B*S, P, token_id_size]

    def compute_predictions(self, beatmap: torch.Tensor,
                condition: torch.Tensor,
                stage: int = -1) -> torch.Tensor:
        """Given a tensor of beatmap of shape [B, S, P] with S the sequence steps and
        P the number of beatmap positions, condition of shape [B, S, dim], return the logits with shape [B, S, P, token_id_size].

        """
        B, S, P  = beatmap.shape
        beatmap = beatmap.view(B*S,P)
        beatmap = beatmap[:,:-1]
        condition = condition.reshape(B*S,1,-1)
        prepend_special_token = torch.tensor([[self.special_token_id]] * (B*S)).cuda()
        beatmap = torch.cat((prepend_special_token, beatmap), dim=1)
        model = self if self._fsdp is None else self._fsdp
        logits = model(beatmap,condition)
        logits = logits.view(B, S, P, -1)
        return logits

    def _sample_next_token(self,
                           sequence: torch.Tensor,
                           conditions: torch.Tensor,
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
        B = sequence.shape[0]
        
        model = self if self._fsdp is None else self._fsdp
        
        logits = model(
            sequence,
            conditions) # B*S,1, token_id_size
        
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
        next_token = next_token.view(B,-1)
        return next_token

    @torch.no_grad()
    def generate(self,
                 conditions: torch.Tensor,
                 prompt: tp.Optional[torch.Tensor] = None,
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
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
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be performed in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (torch.Tensor, optional): Prompt tokens of shape [B, K, T].
            conditions_tensors (list of ConditioningAttributes, optional): List of conditions.
            num_samples (int, optional): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coeff (float, optional): Classifier-free guidance coefficient.
            two_step_cfg (bool, optional): Whether to perform classifier-free guidance with two steps generation.
            remove_prompts (bool): Whether to remove prompts from generation or not.
            check (bool): Whether to apply further checks on generated sequence.
            callback (Callback, optional): Callback function to report generation progress.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        B, S, _ = conditions.shape
        conditions = conditions.reshape(B*S,1,-1)
        unknown_token = -1
        prepend_special_token = torch.tensor([[self.special_token_id]] * (B*S)).cuda()
        gen_codes = torch.full((B*S, max_gen_len), unknown_token, dtype=torch.long, device=device)
        gen_sequence = torch.cat((prepend_special_token, gen_codes), dim=1)

        start_offset_sequence = 1
        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B*S, P]
            for offset in range(start_offset_sequence, gen_sequence_len):
                # get current sequence (note that the streaming API is providing the caching over previous offsets)
                curr_sequence = gen_sequence[..., prev_offset:offset]
                # sample next token from the model, next token and curr_sequence shape is [B*S, 1]
                next_token = self._sample_next_token(
                    curr_sequence,conditions, unconditional_state, use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef, two_step_cfg=two_step_cfg)
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
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes = gen_sequence[:,1:]

        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        
        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.token_id_size).all()
        out_codes = out_codes.view(B, S, -1)
        return out_codes
