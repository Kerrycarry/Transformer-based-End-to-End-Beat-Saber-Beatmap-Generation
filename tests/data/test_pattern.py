from audiocraft import models
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

def plot(mask):
    n1, n2 = mask.shape
    # mask = mask.reshape(n, n)
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="Blues", interpolation="none")
    plt.colorbar(label="Mask Value")
    plt.title(f"{n1}x{n2} Mask Matrix")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.xticks(range(n2))
    plt.yticks(range(n1 - 1, -1, -1))  # Reverse the y-axis tick labels
    plt.show()
n_q = 12
range_n_q = list(range(n_q))
# choice = 'delay'
# choice = 'unroll'
choice = 'parallel'
keep_only_valid_steps=True # training
# keep_only_valid_steps=False # inference
codebooks_pattern_cfg = {'modeling': choice,
                        'delay': {'delays': range_n_q, 'flatten_first': 0, 'empty_initial': 0},
                        'unroll': {'flattening': range_n_q, 'delays': [0, 0, 0, 0]}, 
                        'music_lm': {'group_by': 2},
                        'coarse_first': {'delays': [0, 0, 0]}}
codebooks_pattern_cfg = OmegaConf.create(codebooks_pattern_cfg)

pattern_provider = models.builders.get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)
T = 10
B = 1
card = 2048
special_token_id = 0
codes = torch.ones([B, n_q, T], dtype = torch.int64)

pattern = pattern_provider.get_pattern(T)

sequence, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, special_token_id, keep_only_valid_steps=keep_only_valid_steps
        )
print(sequence)
print(sequence.shape)

logits = torch.rand([B, n_q, sequence.shape[2], card])
origin_logit = logits
logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
    logits, float('nan'), keep_only_valid_steps=keep_only_valid_steps
)
logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
print(logits.shape)
print(origin_logit[:, :, :-1].shape)

assert torch.equal(origin_logit[:, :, :-1], logits)

# plot(sequence.squeeze(0))
# run command:
# python -m tests.data.test_pattern
