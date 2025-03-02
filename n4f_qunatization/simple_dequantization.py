import torch
torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : True,
    "triton.cudagraphs" : False,
    "debug"             : False
}

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def quantize_tensor(x_fp32):
    # Scale to [-1, 1] -> map to int 8 range on [-127, 127] -> round -> cast
    absmax = torch.max(torch.abs(x_fp32))
    c = 127.0 / absmax
    x_int8 = torch.round(c * x_fp32).to(torch.int8)
    return x_int8, c

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def dequantize_tensor(x_int8, c):
    x_fp32 = x_int8.to(torch.float32) / c
    return x_fp32

if __name__ == "__main__":
    
    N = 128
    X = torch.randn(N, N, dtype=torch.float32)

    X_qauntized, c = quantize_tensor(X)
    X_deqauntized = dequantize_tensor(X_qauntized, c)
    
    print(X)
    print(c)
    print(X_qauntized)
    print(X_deqauntized)
    
    
    