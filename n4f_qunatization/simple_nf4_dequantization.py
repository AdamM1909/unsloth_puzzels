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

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options, disable=disable)
def quantize_nf4_blockwise(w_fp32, blocksize=64, absmax_blocksize=256):
    
    def _quantize(x_fp32, blocksize):
        
        # Zero pad to a multiple of blocksize
        x_fp32 = torch.cat([x_fp32.view(-1), torch.zeros((-x_fp32.numel() % blocksize))])

        # Scale block-by-block.
        x_fp32_scaled = (x_fp32.view(-1, blocksize) / (absmax := torch.max(torch.abs(x_fp32.view(-1, blocksize)), dim=-1, keepdim=True)[0])).view(-1)

        # Find closest NF4 grid point index to repesent each value in the nf4 tensor.
        idx = torch.argmin(torch.abs((x_fp32_scaled.unsqueeze(-1) - NF4_GRID)), dim=-1)

        # View in pairs ready to pack 2 uint4 into uint8.
        idx = idx.view(-1, 2)

        # First value of pair goes in the lower 4 bits, second value in the upper 4 bits by shifting << 4 (*16).
        # combining with bitwise OR. 
        x_nf4 = (idx[:, 1] | (idx[:, 0] << 4)).to(torch.uint8)

        return x_nf4, absmax
    
    # Qunatize the weights.
    w_nf4, absmax_fp32 = _quantize(w_fp32, blocksize)
    
    # Qunatize the absmax used in the qunatization of the weights.
    # First scale the absmax to be zero mean, which is the assumption we are making when using NF4.
    absmax_fp32 = absmax_fp32 - (absmax_offset := absmax_fp32.mean())
    # Use the same quantization function to make things simple.
    absmax_nf4, absmax_absmax_fp32 = _quantize(absmax_fp32, absmax_blocksize)
    
    return w_nf4, absmax_nf4, absmax_absmax_fp32, absmax_offset
    

@torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options, disable=disable)
def dequantize_nf4_blockwise(w_nf4, w_shape, absmax_nf4, absmax_absmax_fp32, absmax_offset, blocksize=64, absmax_blocksize=256, dtype=torch.float32):
    
    def _dequantize(x_nf4, absmax, x_shape, blocksize):
        
        # Make an empty tensor to unpack idxs of NF4_GRID back into. 
        idx = torch.empty_like(x_nf4.unsqueeze(1).expand(-1, 2), requires_grad=False, dtype=torch.int64)

        # Unpack lower and upepr 4 bits by leverging an & with 00001111. 
        idx[:, 0], idx[:, 1] = (x_nf4 >> 4) & 0x0F, x_nf4 & 0x0F
        
        # Convert indices back to grid values
        x_fp32 = (NF4_GRID[idx].view(-1, blocksize) * absmax).view(-1).to(dtype)
        
        # If we had to add padding remove it now and get back to original tensor shape.
        x_fp32 = x_fp32[:x_fp32.numel() - (-x_shape.numel() % blocksize)].view(*x_shape)
        
        return x_fp32
    
    print(f"{w_nf4.shape=}")
    print(f"{w_shape=}")
    print(f"{absmax_nf4.shape=}")

    
    # Dequnatize the absmax. There is one absmax for each block in the qunatization.
    absmax_fp32 = _dequantize(absmax_nf4, absmax_absmax_fp32, torch.Size([w_nf4.numel()*2 // blocksize, 1]), absmax_blocksize)
    
    # Not forgetting to add the offset back.
    absmax_fp32  = absmax_fp32 + absmax_offset
    
    # Dequnatize the weights.
    w_fp32 = _dequantize(w_nf4, absmax_fp32, w_shape, blocksize)
    
    return w_fp32


if __name__ == "__main__":
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/b8223fed8aa3f6422f2426828f358f760e208a52/bitsandbytes/functional.py#L1076
    # https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py
    # Here are the kernels in C: https://github.com/bitsandbytes-foundation/bitsandbytes/tree/main/csrc
    
    # There is a mistake with these N, M when comparing to Linear4bit
    # it comes from the qunatization of the absmax. I might not be using the correct method ...
    # bitsnbytes has absmax_nf4.shape=torch.Size([262144]) where i have absmax_nf4.shape=torch.Size([131072])
    # dynamic map type used as code not passed as "nf4" in bitsnbytes
    
    torch.random.manual_seed(0)
    N, M = 2048,  8192
    W = torch.randn(N, , dtype=torch.float32)
    
    # Quantize to NF4
    blocksize, absmax_blocksize = 64, 256
    w_nf4, absmax_nf4, absmax_absmax_fp32, absmax_offset = quantize_nf4_blockwise(W, blocksize, absmax_blocksize)
    
    # Dequantize back to FP32
    W_deqaunt = dequantize_nf4_blockwise(w_nf4, W.shape, absmax_nf4, absmax_absmax_fp32, absmax_offset, blocksize, absmax_blocksize)
    
    print(f"{W_deqaunt=}")
    # print(f"{absmax_nf4=}")
    print(f"{W=}")

