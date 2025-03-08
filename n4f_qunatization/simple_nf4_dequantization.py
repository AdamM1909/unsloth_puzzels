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

# @torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options, disable=disable)
def quantize_nf4(x_fp32):
    
    # Scale tensor to [-1, 1] range. 
    # TODO: Turns out the abs max is also optionally qunatized in bnb.
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/d8d157f4a7708967b63b56d312749fabd21445c2/bitsandbytes/functional.py#L1352C4-L1356C36
    # compress_statistics = False, https://github.com/bitsandbytes-foundation/bitsandbytes/blob/d8d157f4a7708967b63b56d312749fabd21445c2/bitsandbytes/functional.py#L1185C9-L1185C102
    x_fp32_scaled= x_fp32 / (absmax := torch.max(torch.abs(x_fp32)))
    
    # Find closest NF4 grid point index to repesent each value in the nf4 tensor, flatten.
    idx = torch.argmin(torch.abs((x_fp32_scaled.view(-1).unsqueeze(-1) - NF4_GRID)), dim=-1)
  
    # Zero pad to an even number of elements.
    idx = torch.cat([idx, torch.zeros((idx.numel() % 2), dtype=idx.dtype, device=idx.device)])

    # View in pairs ready to pack 2 uint4 into uint8.
    idx = idx.view(-1, 2)
  
    # First value of pair goes in the lower 4 bits, second value in the upper 4 bits by shifting << 4 (*16).
    # combining with bitwise OR. 
    x_nf4 = (idx[:, 1] | (idx[:, 0] << 4)).to(torch.uint8)

    return x_nf4, absmax

# @torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options, disable=disable)
def dequantize_nf4(x_nf4, absmax, x_shape):
    
    # Make an empty tensor to unpack idxs of NF4_GRID back into. 
    idx = torch.empty(x_nf4.numel() * 2, dtype=torch.int64, device=x_nf4.device, requires_grad=False).view(-1, 2)
    
    print(idx.shape)
    print(x_nf4.shape)
    # Unpack lower and upepr 4 bits by leverging an & with 00001111. 
    idx[:, 0], idx[:, 1] = (x_nf4 >> 4) & 0x0F, x_nf4 & 0x0F
    
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
    # TODO: blockwize dequnatization, signature to match fast_dequantize
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/bitsandbytes/functional.py#L958
    
    # Here are the kernels in C: https://github.com/bitsandbytes-foundation/bitsandbytes/tree/main/csrc
    torch.random.manual_seed(0)
    N = 3
    X = torch.randn(N, N, dtype=torch.float32)
    
    # Quantize to NF4
    X_nf4, c = quantize_nf4(X)
    
    # import time 
    # def bench(f, name=None, iters=100, warmup=5, profile=False):
    #     for _ in range(warmup): 
    #         f()
    #     if profile:
    #         with torch.profiler.profile() as prof:
    #             f()
    #         prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

   
    #     begin = time.time()
    #     for _ in range(iters):
    #         f()

    #     res = f"{f'{name}:' if name else ''} {(time.time()-begin)*1e6/iters: .2f}us"
    #     print(res)
        
    # compiled_dequant = torch.compile(dequantize_nf4, fullgraph=True, dynamic=True, options=torch_compile_options)
    # bench(lambda: compiled_dequant(X_nf4, c, X.shape), name='compiled')
    # bench(lambda: dequantize_nf4(X_nf4, c, X.shape), name='normal')
    
    
    # Dequantize back to FP32
    X_dequant = dequantize_nf4(X_nf4, c, X.shape)
    
    # Print results
    print(X)
    print(c)
    print(X_nf4)
    print(X_dequant)
