import torch
import torch._inductor.config
import time
BACKEND = torch.mps
DEVICE = torch.device("mps")

def bench(f, name=None, iters=100, warmup=5, profile=False):
    for _ in range(warmup): 
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    BACKEND.synchronize()
    begin = time.time()
    for _ in range(iters):
        f()
    BACKEND.synchronize()

    res = f"{f'{name}:' if name else ''} {(time.time()-begin)*1e6/iters: .2f}us"
    print(res)


if __name__ == "__main__":
    
    
    def matmul(A, B):
        return A @ B
    
    A, B = torch.rand((1024, 1024), device=DEVICE), torch.rand((1024, 1024), device=DEVICE)
    
    # No real support for classic compilation with MPS
    mps_compiled = torch.compile(matmul, backend="aot_eager")

    bench(lambda: matmul(A, B), name='mps')
    bench(lambda: mps_compiled(A, B), name='compiled')
    
    
    def f1(a, b, c, d):
        a = a.relu()
        b = b.tanh()
        e = a * b
        f = (c + 2).cos()
        return (e + f) * d
    
    inp = [torch.randn(2**24, device=DEVICE) for _ in range(4)]
    f = f1
    nf = torch.compile(f, backend="aot_eager")
    bench(lambda: f(*inp), name="eager")
    bench(lambda: nf(*inp), name="PT 2.0")
    

    
    
    # inp = [torch.randn(2**24, device='cuda') for _ in range(4)]

    # f = f1
    # nf = torch.compile(f)
    # bench(lambda: f(*inp), name="eager")
    # bench(lambda: nf(*inp), name="PT 2.0")
