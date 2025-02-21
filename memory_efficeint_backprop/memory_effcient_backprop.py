import torch
import torch.nn as nn 
import warnings

class MemoryEfficientReduction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, X, linear, labels, reduce_function, chunk_size):
        
        # Set the chunk_size = 1 if there is no reduction taking place to at least maintain perfomance.
        if (reduction := getattr(reduce_function, "reduction")) == "none":
            chunk_size = 1; warnings.warn("reduction of reduction fucntion is None, no VRAM can be saved. Continuing with chunk_size=1.")
        
        # Save tensor and non-tensor inputs in ctx.
        ctx.save_for_backward(X, labels); ctx.inputs = [linear, reduce_function, reduction, chunk_size]
        
        # Chunk the forward pass, using linearity to reduce the chunk losess. Do not calculate/store gradients. 
        with torch.no_grad():
            return getattr(torch, reduction, torch.eye)(torch.cat([reduce_function((_wX := linear(_X).float()).view(-1, _wX.shape[-1]), _labels.view(-1)).view(-1) for _X, _labels in zip(torch.chunk(X, chunk_size, dim=0), torch.chunk(labels, chunk_size, dim=0))]))
    
    @staticmethod
    def backward(ctx, dY):
        
        # Retreive data stored in ctx.
        X, labels = ctx.saved_tensors; linear, reduce_function, reduction, chunk_size = ctx.inputs
        
        # Recompute the forward pass to calculate the gradients.
        dX = []
        for _X, _labels in zip(torch.chunk(X, chunk_size, dim=0), torch.chunk(labels, chunk_size, dim=0)):
            
            # Create a new detached subgraph for each input chunk.
            _X = _X.detach().requires_grad_()
            
            # Enabling gradient computation this time.
            with torch.enable_grad():
                _loss = reduce_function((_wX := linear(_X).float()).view(-1, _wX.shape[-1]), _labels.view(-1)).view(-1) 
                
                # Weight the loss by 1 / chunk_size if this is a "mean" reduction. 
                if reduction == "mean":
                    _loss /= chunk_size
            
            # Accumulate gradients in the linear + save the input chunk gradients.
            _loss.backward(); dX.append(_X.grad)
        
        # Apply upstream gradients to the input gradients.    
        dX = torch.cat(dX, axis=0)*dY
 
        return dX, None, None, None, None
        

if __name__ == "__main__":

    def standard_reduction(batch, linear, labels, reduce_function):
        # Up projection to large space and reduce.
        return  reduce_function((x := linear(batch).float()).view(-1, x.shape[-1]), labels.view(-1))
    
    # Test data.
    torch.manual_seed(0)
    b, q_len, d_h, d_vocab = 128, 4096, 2**5, 2**15
    X, labels, linear = torch.randn(b, d_h, requires_grad=True), torch.randint(0, d_vocab, (b,)), nn.Linear(d_h, d_vocab)
    reduce_function = torch.nn.CrossEntropyLoss(reduction = "mean")

    
    # Standard approach.
    out = standard_reduction(X, linear, labels, reduce_function)
    out.backward()
    grad_X, grad_W = X.grad.clone(), linear.weight.grad.clone()
   
    linear.zero_grad(); X.grad = None

    # Efficient approach.
    chunk_size = 4
    eff_out = MemoryEfficientReduction.apply(X, linear, labels, torch.nn.CrossEntropyLoss(reduction = "mean"), chunk_size)
    eff_out.backward()
    eff_grad_X, eff_grad_W = X.grad.clone(), linear.weight.grad.clone()
   
    
    torch.testing.assert_close(eff_grad_X, grad_X)
    torch.testing.assert_close(eff_grad_W, grad_W)