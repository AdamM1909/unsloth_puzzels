import torch, warnings
import torch.nn as nn 

def standard_reduction(batch, linear, labels, reduce_function):
        # Up projection to large space and reduce.
        return  reduce_function((x := linear(batch).float()).view(-1, x.shape[-1]), labels.view(-1))

class MemoryEfficientReduction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, X, linear, labels, reduce_function, chunk_size):
        
         # Set the chunk_size = 1 if there is no reduction taking place to at least maintain perfomance.
        if (reduction := getattr(reduce_function, "reduction")) == "none":
            chunk_size = 1; warnings.warn("Reduction in reduction function is 'none' so no VRAM can be saved. Continuing with chunk_size=1.")
        
        # Save tensor and non-tensor inputs in ctx.
        ctx.save_for_backward(X, labels); ctx.inputs = [linear, reduce_function, reduction, chunk_size]
        
        # Merge the batch and query length dimension to allow increase batching.
        X, labels= X.view(-1, X.shape[-1]), labels.view(-1)
        
        # Require each batch to be of the same size for later gradient accumulation. 
        assert X.shape[0] % chunk_size == 0, "chunk_size must be a multiple of the batch size or query length."  
        
        # Chunk the forward pass, using linearity to reduce the chunk losess. Do not calculate/store gradients.
        with torch.no_grad():
            return getattr(torch, reduction, torch.eye)(torch.cat([reduce_function(linear(_X).float(), _labels).view(-1) for _X, _labels in zip(torch.chunk(X, chunk_size, dim=0), torch.chunk(labels, chunk_size, dim=0))]))
    
    @staticmethod
    def backward(ctx, dY):
        
        # Retreive data stored in ctx.
        X, labels = ctx.saved_tensors; linear, reduce_function, reduction, chunk_size = ctx.inputs
        
        # Recompute the forward pass to calculate the gradients.
        dX = []
        for _X, _labels in zip(torch.chunk(X.view(-1, X.shape[-1]), chunk_size, dim=0), torch.chunk(labels.view(-1), chunk_size, dim=0)):
            
            # Create a new detached subgraph for each input chunk.
            _X = _X.detach().requires_grad_()
            
            # Enabling gradient computation this time.
            with torch.enable_grad():
                _loss = reduce_function(linear(_X).float(), _labels).view(-1) 
                
                # Weight the loss by 1 / chunk_size if this is a "mean" reduction. 
                if reduction == "mean":
                    _loss /= chunk_size
            
            # Accumulate gradients in the linear + save the input chunk gradients.
            _loss.backward(); dX.append(_X.grad)
        
        # Apply upstream gradients to the input gradients, and view as original input shape. 
        dX = (torch.cat(dX, axis=0)*dY).view(*X.shape)
        
        return dX, None, None, None, None


if __name__ == "__main__":
    
    b = 4
    d_h = 4096
    q_len = 4096 
    d_vocab = 128256 // 16 

    chunk_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.randn(b, q_len, d_h, requires_grad=True, device=device)
    linear = nn.Linear(d_h, d_vocab, device=device)
    
    for reduce_function, labels in [
        (torch.nn.CrossEntropyLoss(reduction="mean"), torch.randint(0, d_vocab, (b, q_len), device=device)),
        ]:

        # Standard approach.
        linear.zero_grad(); X.grad = None
        out = standard_reduction(X, linear, labels, reduce_function)
        out.backward()
        grad_X, grad_W = X.grad.clone(), linear.weight.grad.clone()


        # Efficient approach.
        linear.zero_grad(); X.grad = None
        eff_out = MemoryEfficientReduction.apply(X, linear, labels, reduce_function, chunk_size)
        eff_out.backward()
        eff_grad_X, eff_grad_W = X.grad.clone(), linear.weight.grad.clone()

        # Test
        torch.testing.assert_close(out, eff_out)
        torch.testing.assert_close(eff_grad_X, grad_X)
        torch.testing.assert_close(eff_grad_W, grad_W)
        
        print(f'Passed: {reduce_function}')