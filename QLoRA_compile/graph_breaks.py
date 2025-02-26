import logging
import torch
import os

os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1" 
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : True,
    "triton.cudagraphs" : False,
}

torch._inductor.config.debug = True
torch._logging.set_logs(
    dynamo = logging.WARN,
    inductor = logging.WARN,
    graph_breaks = True,
    recompiles = True, # if a guard is triggered (checks for shape changes, datatypes, device or control flow changes) then torch recompiles the function.
    recompiles_verbose = True,
    compiled_autograd_verbose = True,
    # aot_joint_graph = True, # Enable for more logs
    # aot_graphs = True,
)
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False 


if __name__ == "__main__":
    
    @torch.compile(fullgraph=False)
    def function_with_graph_break(x):
        y = x + 1
        print("Graph break")  
        z = y * 2  
        return z

    result = function_with_graph_break(torch.randn(3, 3))
    
    @torch.compile
    def fn(x):
        return x + 1

    fn(torch.ones(3, 3))
    
    import torch._dynamo as dynamo
    def toy_example(a, b):
        x = a / (torch.abs(a) + 1)
        print("woo")
        if b.sum() < 0:
            b = b * -1
        return x * b
    explanation = dynamo.explain(toy_example)(torch.randn(10), torch.randn(10))
    
    print(explanation)