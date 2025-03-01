import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial
DEVICE = torch.device("cuda")

B, H, S, D = 1, 4, 256, 64

dropout_prob = 0.1
full_dropout = bool_mask = (torch.rand((B, H, S, D), device=DEVICE) > dropout_prob)

def dropout(score, b, h, q_idz, kv_idx):
    return torch.where(full_dropout[b, h, q_idz, kv_idx], -float("inf"), score)


# dropout_mask = torch.rand((seq_len := query_states.size(-2), seq_len), device=query_states.device)
    # def dropout(q_idx, kv_idx):
    # return dropout_mask[q_idx, kv_idx] > dropout
        # def dropout(q_idx, kv_idx):
    # return dropout_mask[q_idx, kv_idx] > dropout
    # def causal_drop(b, h, q_idx, kv_idx): return causal(q_idx, kv_idx) #& dropout(q_idx, kv_idx)


if __name__ == "__main__":
    # From  https://github.com/pytorch-labs/attention-gym/issues/77
    make_tensor = partial(torch.randn, (B, H, S, D), device=DEVICE, dtype=torch.float16, requires_grad=True)
    print(full_dropout)
    query, key, value = make_tensor(), make_tensor(), make_tensor()
    compiled_flex = torch.compile(flex_attention, fullgraph=True)
    out = compiled_flex(query, key, value, score_mod=dropout)
    # print(out)