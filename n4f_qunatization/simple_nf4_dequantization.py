import torch

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : True,
    "triton.cudagraphs" : False,
    "debug"             : False
}
disable = True
DEBUG = 0
NF4_GRID = torch.tensor([
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ]
)

@torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options, disable=disable)
def quantize_nf4(x_fp32):
    
    # Scale tensor to [-1, 1] range
    x_fp32_scaled = x_fp32 / (absmax := torch.max(torch.abs(x_fp32)))
    if DEBUG: print(f"{x_fp32_scaled=}")
    
    # Find closest NF4 grid point index to repesent each value in the nf4 tensor, flatten.
    idx = torch.argmin(torch.abs((x_fp32_scaled.view(-1).unsqueeze(-1) - NF4_GRID)), dim=-1)
  
    # Zero pad to an even number of elements.
    idx = torch.cat([idx, torch.zeros((idx.numel() % 2), dtype=idx.dtype, device=idx.device)])

    # View in pairs ready to pack 2 uint4 into uint8.
    idx = idx.view(-1, 2)
  
    # First value of pair goes in the lower 4 bits, second value in the upper 4 bits by shifting << 4 (*16).
    # combining with bitwise OR. 
    x_nf4 = (idx[:, 0] | (idx[:, 1] << 4)).to(torch.uint8)

    return x_nf4, absmax

@torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options, disable=disable)
def dequantize_nf4(x_nf4, absmax, x_shape):
    
    # Make an empty tensor to unpack idxs of NF4_GRID back into. 
    idx = torch.empty(x_nf4.numel() * 2, dtype=torch.int64, device=x_nf4.device).view(-1, 2)
    
    # Unpack lower and upepr 4 bits by leverging an & with 00001111. 
    idx[:, 0], idx[:, 1] = x_nf4 & 0x0F, (x_nf4 >> 4) & 0x0F
    
    # If we had to add padding remove it now and get back to original tensor shape.
    idx = idx.view(-1)[:idx.numel() - (x_shape.numel() % 2)].view(*x_shape)
    
    # Convert indices back to grid values
    x_fp32 = NF4_GRID[idx]

    # Scale back to original scale
    x_fp32 = x_fp32 * absmax
    
    return x_fp32

if __name__ == "__main__":
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/b8223fed8aa3f6422f2426828f358f760e208a52/bitsandbytes/functional.py#L1076
    # https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py
    torch.random.manual_seed(0)
    N = 127
    X = torch.randn(N, N, dtype=torch.float32)
    
    # Quantize to NF4
    X_nf4, c = quantize_nf4(X)
    
    # Dequantize back to FP32
    X_dequant = dequantize_nf4(X_nf4, c, X.shape)
    
    # Print results
    print(X[:5, :5])
    print(c)
    print(X_nf4[:5])
    print(X_dequant[:5, :5])
    
    # TODO: blockwize dequnatization, signature to match fast_dequantize
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/bitsandbytes/functional.py#L958