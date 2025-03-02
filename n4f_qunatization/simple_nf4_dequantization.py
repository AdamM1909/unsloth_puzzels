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

# https://github.com/bitsandbytes-foundation/bitsandbytes/blob/b8223fed8aa3f6422f2426828f358f760e208a52/bitsandbytes/functional.py#L1076
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

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options, disable=disable)
def quantize_nf4(x_fp32):

    # Scale tensor to [-1, 1] range
    x_fp32_scaled = x_fp32 / (absmax := torch.max(torch.abs(x_fp32)))
    
    # Find closest NF4 grid point to repesent each value in the qunatized tensor
    d = torch.abs((x_fp32_scaled.unsqueeze(-1) - NF4_GRID.view(1, 1, -1)))
    x_nf4 = torch.argmin(d, dim=-1)
    
    # Store indices as uint8 (note: in production, you'd pack two 4-bit values per byte)
    x_nf4 = x_nf4 #.to(torch.uint8)
    
    return x_nf4, absmax

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options, disable=disable)
def dequantize_nf4(x_nf4, absmax):
    # Convert indices back to grid values
    x_dequant = NF4_GRID[x_nf4]
    
    # Scale back to original range
    x_fp32 = x_dequant * absmax
    
    return x_fp32

if __name__ == "__main__":
    # https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py
    torch.random.manual_seed(0)
    N = 128
    X = torch.randn(N, N, dtype=torch.float32)
    
    # Quantize to NF4
    X_nf4, c = quantize_nf4(X)
    
    # Dequantize back to FP32
    X_dequant = dequantize_nf4(X_nf4, c)
    
    # Print results
    print(X[:5, :5])
    print(c)
    print(X_nf4[:5, :5])
    print(X_dequant[:5, :5])