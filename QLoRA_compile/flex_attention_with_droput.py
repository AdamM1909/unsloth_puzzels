import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial
DEVICE = torch.device("cuda")


def dropout(score, b, h, q_idz, kv_idx):
    return torch.where((torch.rand((B, H, S, D), device=DEVICE) > dropout_prob)[b, h, q_idz, kv_idx], -float("inf"), score)


if __name__ == "__main__":
    # From  https://github.com/pytorch-labs/attention-gym/issues/77
    
    B, H, S, D = 1, 4, 256, 64
    dropout_prob = 0.1
    make_tensor = partial(torch.randn, (B, H, S, D), device=DEVICE, dtype=torch.float16, requires_grad=True)
    query, key, value = make_tensor(), make_tensor(), make_tensor()
    compiled_flex = torch.compile(flex_attention, fullgraph=True)
    out = compiled_flex(query, key, value, score_mod=dropout)
 