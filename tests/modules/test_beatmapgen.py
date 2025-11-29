from itertools import product
import typing as tp
# import pytest
import torch

from audiocraft.modules.transformer import (
    StreamingMultiheadAttention, StreamingTransformer, set_efficient_attention_backend)


def get_mask_transfer_lm(causal, segment_duration, autocast: bool = True, local_cross_attention= True, position_size = 12, local_self_attention= False, ca_window_size=0, sa_window_size=4) -> tp.Optional[torch.Tensor]:
    # N = position_size
    # query_length = query.shape[1]
    # if self.block_self_attention:
    #     assert query_length % position_size == 0
    # beatmapgen inference case
    # if self._is_streaming:
    #     attn_mask = None
    # beatmapgen sa training case
    if causal:
        if True:
            # custom_attn_mask = True
            # if not local_self_attention:
            query_len = segment_duration * position_size
            key_len = segment_duration * position_size
            d = torch.arange(segment_duration, device = 'cuda').repeat_interleave(position_size)
            dT = torch.arange(segment_duration, device = 'cuda').repeat_interleave(position_size)
            mask_op = ">="
            # else:
                # add full block attention along diagonal line and tringular mask within block size
                # raise RuntimeError("Not supported at the moment")
                # attn_mask = torch.zeros(1, 1, query_length, query_length, dtype=torch.bool)
                # for i in range(0, query_length, N):
                #     index = i - N * sa_window_size
                #     if index < 0:
                #         index = 0
                #     attn_mask[..., i:i+N, index:i+N] = True
        # else:
        #     if not local_self_attention:
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
                #     index = i - sa_window_size
                #     if index < 0:
                #         index = 0
                #     attn_mask[..., i, index:i+1] = True
    # beatmapgen ca training case
    else:
        if not local_cross_attention: # full cross attention case
            raise RuntimeError("Not supported at the moment")
            # attn_mask = None
        else:
            
            # if self.block_self_attention:
            #     assert (query_length // position_size) == key.shape[1]
            #     assert (query_length // position_size) == value.shape[1]
            # else:
            #     assert query_length == key.shape[1]
            #     assert query_length == value.shape[1]
            # cross_src_length = key.shape[1]
            # custom_attn_mask = True
            # if self.block_cross_attention or not self.block_self_attention:
            #     raise RuntimeError("Not supported at the moment")
            #     # attn_mask = torch.eye(cross_src_length, cross_src_length, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
            # else:
            query_len = segment_duration * position_size
            d = torch.arange(segment_duration, device = 'cuda').repeat_interleave(position_size)
            windows = ca_window_size*2 + 1
            dT = torch.arange(segment_duration, device = 'cuda').repeat_interleave(windows)
            key_len = segment_duration * windows
            mask_op = "=="
    d = d.view(1, -1, 1)
    dT = dT.view(1, -1, 1).transpose(1, 2)
    if mask_op == ">=":
        if local_self_attention:
            mask = (dT <= d) & ((d - dT) <= sa_window_size)
        else:
            mask = d >= dT
    elif mask_op == "==":
        mask = d == dT
    zero_tensor = torch.full((1,), 0.0, device = 'cuda', dtype = torch.float16)
    neg_inf_tensor = torch.full((1,), float('-inf'), device = 'cuda', dtype = torch.float16)
    attn_mask = torch.where(mask, zero_tensor, neg_inf_tensor).reshape(1, 1, query_len, key_len)
    if not autocast:
        attn_mask = attn_mask.to(dtype=torch.float32)
    return attn_mask



def test_transformer_causal_streaming():
    set_efficient_attention_backend('xformers')
    torch.manual_seed(1234)
    attn_mask_for_sa = get_mask_transfer_lm(segment_duration=4, causal = True, autocast = False).to(device='cuda')
    attn_mask_for_ca = get_mask_transfer_lm(segment_duration=4, causal = False, autocast = False).to(device='cuda')
    # Test that causality and receptive fields are properly handled.
    # looking at the gradients
    blockwise_attention_kwargs = {'use_transfer_lm': True, 'block_self_attention': True, 'local_cross_attention': True, 'local_self_attention': False}
    tr = StreamingTransformer(
        32, 4, 2, position_size = 12, block_self_attention=True, cross_attention=True, memory_efficient= True,
        causal=True,
        dropout=0., blockwise_attention_kwargs = blockwise_attention_kwargs, device='cuda')
    tr.to(device='cuda')
    steps = 4*12
    for k in [0, 5, 11, 16, 23, 47]:
        x = torch.randn(4, steps, 32, requires_grad=True).to(device='cuda')
        cross_attention_input = torch.randn(4, 4, 32, requires_grad=True).to(device='cuda')
        x.retain_grad()
        cross_attention_input.retain_grad()
        y = tr(x, cross_attention_src=cross_attention_input,
                            src_mask=attn_mask_for_sa, cross_src_mask=attn_mask_for_ca)
        y[:, k].abs().sum().backward()
        k_right = (k//12)*12 + 11
        if k < steps:
            # 验证k之后的x是否有back prog
            assert torch.allclose(x.grad[:, k_right + 1:], torch.tensor(0.)), x.grad[:, k_right + 1:].norm()
        assert not torch.allclose(x.grad[:, :k_right + 1], torch.tensor(0.)), x.grad[:, :k_right + 1].norm()

        k_cross = k // 12
        assert not torch.allclose(cross_attention_input.grad[:, :k_cross+1], torch.tensor(0.)), cross_attention_input.grad[:, :k_cross+1].norm()
        assert torch.allclose(cross_attention_input.grad[:, k_cross+1:], torch.tensor(0.)), cross_attention_input.grad[:, k_cross+1:].norm()
    
    # Now check that streaming gives the same result at batch eval.
    x = torch.randn(4, steps, 32).to(device='cuda')
    cross_attention_input = torch.randn(4, 4, 32).to(device='cuda')
    y = tr(x, cross_attention_src=cross_attention_input,
                            src_mask=attn_mask_for_sa, cross_src_mask=attn_mask_for_ca)
    ys = []
    with tr.streaming():
        for k in range(4):
            chunk = x[:, k*12:(k+1)*12, :]
            chunk_cross = cross_attention_input[:,k]
            chunk_cross=chunk_cross.unsqueeze(1)
            ys.append(tr(chunk, cross_attention_src=chunk_cross))
    y_stream = torch.cat(ys, dim=1)
    delta = torch.norm(y_stream - y) / torch.norm(y)
    assert delta < 1e-6, delta

if __name__ == '__main__':
    test_transformer_causal_streaming()