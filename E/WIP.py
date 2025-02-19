import torch
from torch.utils import checkpoint
import torch.nn as nn 
from torch.profiler import profile, record_function, ProfilerActivity


def transformation_function(batch, linear, labels):
        x = linear(batch).float() # Up projection to large space
        from torch.nn import CrossEntropyLoss
        down_projection_function = CrossEntropyLoss(reduction = "mean")
        # Down projection to small space
        loss = down_projection_function(x.view(-1, x.shape[-1]), labels.view(-1))
        return loss

class MemoryEfficientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, labels, forward_function, chunk_size=2):
        ctx.save_for_backward(X, weight, labels)
        ctx.forward_function = forward_function
        ctx.weight = weight
        ctx.chunk_size = chunk_size
        with torch.no_grad():
            return torch.cat([forward_function((_wX := torch.nn.functional.linear(_X, weight).float()).view(-1, _wX.shape[-1]), _labels.view(-1)).view(-1) for _X, _labels in zip(torch.chunk(X, chunk_size, dim=0), torch.chunk(labels, chunk_size, dim=0))]).mean()

    @staticmethod
    def backward(ctx, dY):
        X, weight, labels = ctx.saved_tensors
        forward_function, chunk_size = ctx.forward_function, ctx.chunk_size
 

        total_elements = labels.numel()
        scale = 1 # dY / total_elements

        dX, dW = torch.zeros_like(X), torch.zeros_like(weight)
    
        start = 0
        for _X, _labels in zip(torch.chunk(X, chunk_size, dim=0), torch.chunk(labels, chunk_size, dim=0)):
            _X = _X.detach().requires_grad_()
            with torch.enable_grad():
                _loss = forward_function((_wX := torch.nn.functional.linear(_X, weight).float()).view(-1, _wX.shape[-1]), _labels.view(-1)).view(-1) * scale
    
            _dX, _dW = torch.autograd.grad(_loss, (_X, weight), retain_graph=True)

            dX[start:(end := start + _X.shape[0])] = _dX
            dW += _dW
            start = end
    
        return dX, dW, None, None, None



if __name__ == "__main__":
    # Naive 2*4*4096*128000 / (1024)**3 Gb
    torch.manual_seed(0)
    
    b, q_len, d_h, d_vocab = 4, 4096, 2**5, 2**15
    X = torch.randn(b, d_h, requires_grad=True)
    labels = torch.randint(0, d_vocab, (b,))
    
    """
    Plan
    1) Chunk the forward of the up project i.e. linear over the batch dimension
    2) Save the input and recompute the same chunks for the backward.
    """
    
    # Normal approach
    linear = nn.Linear(d_h, d_vocab)
    normal_out = transformation_function(X, linear, labels)
    normal_out.backward()
    normal_grad_X = X.grad.clone()
    normal_grad_W = linear.weight.grad.clone()


    linear.zero_grad()
    X.grad = None

    # Efficient
    cs = 2
    eff_out = MemoryEfficientFunction.apply(X, linear.weight, labels, torch.nn.CrossEntropyLoss(reduction = "mean"), cs)
    eff_out.backward()
    eff_grad_X = X.grad.clone()
    eff_grad_W = linear.weight.grad.clone()
    
    
    
    print('done')
    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    #    # Normal approach
    #     linear = nn.Linear(d_h, d_vocab)
    #     normal_out = transformation_function(X, linear, labels)
    #     normal_out.backward()
    #     normal_grad_X = X.grad.clone()
    #     normal_grad_W = linear.weight.grad.clone()
        
        
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # prof.export_chrome_trace("trace.json")
        