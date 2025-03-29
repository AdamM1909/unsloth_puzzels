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

# https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e772a9e8723cfc2036fecc830c328ad3b9705250/bitsandbytes/functional.py#L363
UINT8_GRID = torch.tensor([
    -0.992968738079071, -0.9789062738418579, -0.96484375, -0.9507812261581421, 
    -0.936718761920929, -0.922656238079071, -0.9085937738418579, -0.89453125, 
    -0.8804687261581421, -0.866406261920929, -0.852343738079071, -0.8382812738418579, 
    -0.82421875, -0.8101562261581421, -0.796093761920929, -0.782031238079071, 
    -0.7679687738418579, -0.75390625, -0.7398437261581421, -0.725781261920929, 
    -0.7117187976837158, -0.6976562738418579, -0.68359375, -0.6695312261581421, 
    -0.655468761920929, -0.6414062976837158, -0.6273437738418579, -0.61328125, 
    -0.5992187261581421, -0.585156261920929, -0.5710937976837158, -0.5570312738418579, 
    -0.54296875, -0.5289062261581421, -0.5148437023162842, -0.500781238079071, 
    -0.48671871423721313, -0.47265625, -0.4585937261581421, -0.44453126192092896,
    -0.43046873807907104, -0.4164062440395355, -0.40234375, -0.3882812261581421,
    -0.37421876192092896, -0.36015623807907104, -0.3460937440395355, -0.33203125, 
    -0.3179687261581421, -0.30390626192092896, -0.28984373807907104, -0.2757812738418579,
    -0.26171875, -0.24765624105930328, -0.23359374701976776, -0.21953125298023224, 
    -0.20546874403953552, -0.19140625, -0.17734375596046448, -0.16328124701976776, 
    -0.14921875298023224, -0.13515624403953552, -0.12109375, -0.10703125596046448,
    -0.09859374910593033, -0.09578125923871994, -0.09296875447034836, -0.09015624970197678,
    -0.08734375238418579, -0.08453124761581421, -0.08171875774860382, -0.07890625298023224,
    -0.07609374821186066, -0.07328125089406967, -0.07046874612569809, -0.0676562562584877, 
    -0.06484375149011612, -0.062031250447034836, -0.05921875312924385, -0.05640624836087227,
    -0.053593751043081284, -0.05078125, -0.047968748956918716, -0.04515625163912773, 
    -0.04234374687075615, -0.039531249552965164, -0.03671875223517418, -0.033906251192092896, 
    -0.031093750149011612, -0.028281250968575478, -0.025468749925494194, -0.02265625074505806,
    -0.019843751564621925, -0.017031250521540642, -0.014218750409781933, -0.011406250298023224,
    -0.009718749672174454, -0.009156249463558197, -0.008593750186264515, -0.008031249977648258, 
    -0.0074687497690320015, -0.006906250026077032, -0.006343749817460775, -0.005781250074505806, 
    -0.0052187503315508366, -0.0046562496572732925, -0.004093749914318323, -0.0035312497057020664,
    -0.002968749962747097, -0.002406249986961484, -0.001843750011175871, -0.001281249918974936,
    -0.0009437500848434865, -0.0008312499849125743, -0.0007187500596046448, -0.0006062500760890543, 
    -0.000493750034365803, -0.0003812500217463821, -0.0002687500382307917, -0.00015625001105945557, 
    -8.874999912222847e-05, -6.625000241911039e-05, -4.374999844003469e-05, -2.1249998098937795e-05, 
    -7.749999895168003e-06, -3.250000190746505e-06, -5.500000384017767e-07, 0.0, 5.500000384017767e-07, 
    3.250000190746505e-06, 7.749999895168003e-06, 2.1249998098937795e-05, 4.374999844003469e-05, 
    6.625000241911039e-05, 8.874999912222847e-05, 0.00015625001105945557, 0.0002687500382307917,
    0.0003812500217463821, 0.000493750034365803, 0.0006062500760890543, 0.0007187500596046448, 
    0.0008312499849125743, 0.0009437500848434865, 0.001281249918974936, 0.001843750011175871, 
    0.002406249986961484, 0.002968749962747097, 0.0035312497057020664, 0.004093749914318323, 
    0.0046562496572732925, 0.0052187503315508366, 0.005781250074505806, 0.006343749817460775,
    0.006906250026077032, 0.0074687497690320015, 0.008031249977648258, 0.008593750186264515, 
    0.009156249463558197, 0.009718749672174454, 0.011406250298023224, 0.014218750409781933, 
    0.017031250521540642, 0.019843751564621925, 0.02265625074505806, 0.025468749925494194, 
    0.028281250968575478, 0.031093750149011612, 0.033906251192092896, 0.03671875223517418, 
    0.039531249552965164, 0.04234374687075615, 0.04515625163912773, 0.047968748956918716, 
    0.05078125, 0.053593751043081284, 0.05640624836087227, 0.05921875312924385, 
    0.062031250447034836, 0.06484375149011612, 0.0676562562584877, 0.07046874612569809, 
    0.07328125089406967, 0.07609374821186066, 0.07890625298023224, 0.08171875774860382, 
    0.08453124761581421, 0.08734375238418579, 0.09015624970197678, 0.09296875447034836, 
    0.09578125923871994, 0.09859374910593033, 0.10703125596046448, 0.12109375, 
    0.13515624403953552, 0.14921875298023224, 0.16328124701976776, 0.17734375596046448,
    0.19140625, 0.20546874403953552, 0.21953125298023224, 0.23359374701976776, 
    0.24765624105930328, 0.26171875, 0.2757812738418579, 0.28984373807907104, 
    0.30390626192092896, 0.3179687261581421, 0.33203125, 0.3460937440395355, 
    0.36015623807907104, 0.37421876192092896, 0.3882812261581421, 0.40234375, 
    0.4164062440395355, 0.43046873807907104, 0.44453126192092896, 0.4585937261581421,
    0.47265625, 0.48671871423721313, 0.500781238079071, 0.5148437023162842, 
    0.5289062261581421, 0.54296875, 0.5570312738418579, 0.5710937976837158, 
    0.585156261920929, 0.5992187261581421, 0.61328125, 0.6273437738418579, 
    0.6414062976837158, 0.655468761920929, 0.6695312261581421, 0.68359375, 
    0.6976562738418579, 0.7117187976837158, 0.725781261920929, 0.7398437261581421, 
    0.75390625, 0.7679687738418579, 0.782031238079071, 0.796093761920929, 
    0.8101562261581421, 0.82421875, 0.8382812738418579, 0.852343738079071, 
    0.866406261920929, 0.8804687261581421, 0.89453125, 0.9085937738418579, 
    0.922656238079071, 0.936718761920929, 0.9507812261581421, 0.96484375, 
    0.9789062738418579, 0.992968738079071, 1.0]
)

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options, disable=disable)
def quantize_nf4_blockwise(w_fp32, blocksize=64, absmax_blocksize=256):
    
    def _get_qunatization_index(x_fp32, blocksize, qunatization_grid):
        
        # Zero pad to a multiple of blocksize
        x_fp32 = torch.cat([x_fp32.view(-1), torch.zeros((-x_fp32.numel() % blocksize))])

        # Scale block-by-block.
        x_fp32_scaled = (x_fp32.view(-1, blocksize) / (absmax := torch.max(torch.abs(x_fp32.view(-1, blocksize)), dim=-1, keepdim=True)[0])).view(-1)

        # Find closest grid point index to repesent each value in the qunatized tensor.
        idx = torch.argmin(torch.abs((x_fp32_scaled.unsqueeze(-1) - qunatization_grid)), dim=-1)
        
        return idx, absmax
    
    def _quantize_nf4(x_fp32, blocksize):
        
        # Find closest NF4 grid point index to repesent each value in the nf4 tensor.
        idx, absmax = _get_qunatization_index(x_fp32, blocksize, NF4_GRID)

        # View in pairs ready to pack 2 uint4 into uint8.
        idx = idx.view(-1, 2)

        # First value of pair goes in the lower 4 bits, second value in the upper 4 bits by shifting << 4 (*16).
        # combining with bitwise OR. 
        x_nf4 = (idx[:, 1] | (idx[:, 0] << 4)).to(torch.uint8)

        return x_nf4, absmax
    
    def _quantize_uint8(x_fp32, blocksize):
        
        # Find closest uint8 grid point index to repesent each value in the uint8 tensor.
        idx, absmax = _get_qunatization_index(x_fp32, blocksize, UINT8_GRID)
        
        # Cast idx to unit8 and return.
        return idx.to(torch.uint8), absmax
        
    # Qunatize the weights.
    w_nf4, absmax_fp32 = _quantize_nf4(w_fp32, blocksize)
    
    # Qunatize the absmax used in the qunatization of the weights.
    # First scale the absmax to be zero mean, which is the assumption we are making when using NF4.
    absmax_fp32 = absmax_fp32 - (absmax_offset_fp32 := absmax_fp32.mean())
    
    # Now qunatize the absmaz to uint8 using a similar approach.
    absmax_unit8, absmax_absmax_fp32 = _quantize_uint8(absmax_fp32, absmax_blocksize)
    
    return w_nf4, absmax_unit8, absmax_absmax_fp32, absmax_offset_fp32
    

@torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options, disable=disable)
def dequantize_nf4_blockwise(w_nf4, w_shape, absmax_unit8, absmax_absmax_fp32, absmax_offset_fp32, blocksize=64, absmax_blocksize=256, dtype=torch.float32):
    
    def _qunatization_index_to_dtype(idx, absmax, x_shape, blocksize, qunatization_grid):

        # Convert indices back to grid values
        x_dtype = (qunatization_grid[idx].view(-1, blocksize) * absmax).view(-1).to(dtype)
        
        # If we had to add padding remove it now and get back to original tensor shape.
        x_dtype = x_dtype[:x_dtype.numel() - (-x_shape.numel() % blocksize)].view(*x_shape)
        
        return x_dtype

    print(f"{w_nf4.shape=}")
    print(f"{w_shape=}")
    print(f"{absmax_unit8.shape=}")
    
    # Cast to int allowing Torch indexing.
    absmax_idx = absmax_unit8.to(torch.int64)

    # Dequnatize the absmax. There is one absmax for each block in the qunatization.
    absmax_fp32 = _qunatization_index_to_dtype(absmax_idx, absmax_absmax_fp32, torch.empty([w_nf4.numel()*2 // blocksize, 1]).shape, absmax_blocksize, UINT8_GRID)
    
    # Not forgetting to add the offset back.
    absmax_fp32  = absmax_fp32 + absmax_offset_fp32
    
    # Make an empty tensor to unpack weight qunatization idxs into. 
    w_idx = torch.empty_like(w_nf4.unsqueeze(1).expand(-1, 2), requires_grad=False, dtype=torch.int64)

    # Unpack lower and upepr 4 bits, leverging & with 00001111. 
    w_idx[:, 0], w_idx[:, 1] = (w_nf4 >> 4) & 0x0F, w_nf4 & 0x0F
    
    # Dequnatize the weights.
    w_dtype = _qunatization_index_to_dtype(w_idx, absmax_fp32, w_shape, blocksize, NF4_GRID)
    
    return w_dtype

def dequnatize_bnb(weight):
    # Wrapper to dequnatize a Params4bit from bitsandbytes.
    
    # Unpack variables.
    w_nf4, quant_state = weight.weight, weight.quant_state
    absmax_unit8, w_shape, blocksize, dtype, _state2 = quant_state.absmax, quant_state.shape, quant_state.blocksize, quant_state.dtype, quant_state.state2
    absmax_blocksize, absmax_absmax_fp32, absmax_offset_fp32 = _state2.blocksize, _state2.absmax, quant_state.offset
    
    # This is the zero padding on the weights.
    w_nf4 = torch.cat([w_nf4.squeeze(-1), 119*torch.ones((-w_nf4.numel() % (blocksize//2)), dtype=torch.uint8, device=w_nf4.device)])

    # This is the zero padding on the absmax.
    absmax_unit8 = torch.cat([absmax_unit8, 119*torch.ones((-absmax_unit8.numel() % (absmax_blocksize)), dtype=torch.uint8, device=absmax_unit8.device)])    
    
    # Reshape to allign with my functions.
    absmax_absmax_fp32 = absmax_absmax_fp32.unsqueeze(-1)

    return dequantize_nf4_blockwise(w_nf4=w_nf4, 
                                    w_shape=w_shape, 
                                    absmax_unit8=absmax_unit8, 
                                    absmax_absmax_fp32=absmax_absmax_fp32, 
                                    absmax_offset_fp32=absmax_offset_fp32,
                                    blocksize=blocksize, 
                                    absmax_blocksize=absmax_blocksize,
                                    dtype=dtype)
    

if __name__ == "__main__":
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/b8223fed8aa3f6422f2426828f358f760e208a52/bitsandbytes/functional.py#L1076
    # https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py
    # Here are the kernels in C: https://github.com/bitsandbytes-foundation/bitsandbytes/tree/main/csrc
    
    # There is a mistake with these N, M when comparing to Linear4bit
    # it comes from the qunatization of the absmax. I might not be using the correct method ...
    # bitsnbytes has absmax_nf4.shape=torch.Size([262144]) where i have absmax_nf4.shape=torch.Size([131072])
    # dynamic map type used as code not passed as "nf4" in bitsnbytes
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e772a9e8723cfc2036fecc830c328ad3b9705250/bitsandbytes/functional.py#L1248C4-L1268C19
    # They use quant_type="fp4" but this doesnt explain why they have twice as many ? 
    
    # 64,  128 works, 64*2, 128 does not ... same issue
    
    torch.random.manual_seed(0)
    N, M = 64,  128*2
    W = torch.randn(N, M, dtype=torch.float32)
    
    # Quantize to NF4
    blocksize, absmax_blocksize = 64, 256
    w_nf4, absmax_unit8, absmax_absmax_fp32, absmax_offset = quantize_nf4_blockwise(W, blocksize, absmax_blocksize)
    
    # Dequantize back to FP32
    W_deqaunt = dequantize_nf4_blockwise(w_nf4, W.shape, absmax_unit8, absmax_absmax_fp32, absmax_offset, blocksize, absmax_blocksize)
    
    print(f"{W_deqaunt=}")
    print(f"{W=}")

    print('okay')

