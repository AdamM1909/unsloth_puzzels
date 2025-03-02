from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import os
import torch
import transformers.models.llama.modeling_llama

# --------------------------------------- COMPILE CONFIG ---------------------------------------
torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : True,
    "triton.cudagraphs" : False,
    "debug": True
}
# --------------------------------------- LOGGING ---------------------------------------
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1" # Cannot reuse precomputed compilations
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
    "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "0"
import logging
torch._inductor.config.debug = True
torch._logging.set_logs(
    dynamo = logging.WARN,
    inductor = logging.WARN,
    graph_breaks = True,
    recompiles = True, # i.e. shape changes, datatypes, device or control flow changes??
    recompiles_verbose = True,
    compiled_autograd_verbose = True,
    # aot_joint_graph = True, # Enable for more logs
    # aot_graphs = True,
)
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False
# --------------------------------------- MLP ---------------------------------------
@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_llama_mlp(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

transformers.models.llama.modeling_llama.LlamaMLP.forward = compiled_llama_mlp
# --------------------------------------- FLEX ATTENTION ---------------------------------------
# This is a WIP ... maybe easier to do the roatary embeddings as well...
# block_mask will not compile 
# struggling to add dropout to block_mask and leverage sparsity further...

# https://github.com/pytorch/pytorch/blob/e49c0acc396e89baf8c6450e1fa0571d4ce2d4ed/torch/nn/attention/flex_attention.py#L594
# Use https://pytorch.org/blog/flexattention/#causal-mask-1
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_flex_attention_interface(self, query_states, key_states, value_states, attention_mask=None, dropout=0.0, scaling=None, **kwargs):

    # def get_block_mask(seq_len=4096):
    #   def causal(b, h, q_idx, kv_idx): return q_idx >= kv_idx
    #   return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

    # # Leverage the sparsity by using block_mask
    # block_mask = get_block_mask(query_states.size(-2))

    # Call flex_attention
    attn_output = flex_attention(
        query_states,
        key_states,
        value_states,
        # block_mask=block_mask,
        scale=scaling,
        enable_gqa=True,
    )

    return attn_output, None


transformers.models.llama.modeling_llama.ALL_ATTENTION_FUNCTIONS.update(
    {
        "compiled_flex": compiled_flex_attention_interface,
    }
)

# --------------------------------------- Layer Norm ---------------------------------------

@torch.compile(fullgraph=False, dynamic=True, options=torch_compile_options)
def compiled_layernorm(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = compiled_layernorm

# --------------------------------------- QUANTIZATION KERNEL FOR QLORA ---------------------------------------
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
from unsloth.kernels.utils import fast_dequantize

# https://github.com/huggingface/peft/blob/6c44096c7b8d55a2ecf24be9bc68393467e1584a/src/peft/tuners/lora.py#L1177C26-L1177C54

# Create a compiled dequantization function
@torch.compile(fullgraph=False)
def compiled_dequantize(weight_obj):
    return fast_dequantize(weight_obj.weight, weight_obj.weight.quant_state)

# Create a forward function for bitsandbytes Linear4bit that uses compiled dequantization
@torch.compile(fullgraph=False)
def compiled_linear4bit_forward(self, x):
    return torch.nn.functional.linear(x, compiled_dequantize(self), self.bias)

bnb.nn.Linear4bit.forward = compiled_linear4bit_forward

# --------------------------------------- QLORA FOWARD ---------------------------------------
from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit

# Create a forward function for PEFT LoRA Linear4bit
@torch.compile(fullgraph=False)
def compiled_lora_linear4bit_forward(self, x):
    # Check if adapters are disabled
    base_output = compiled_linear4bit_forward(self.base_layer, x)
    if self.disable_adapters:
        return base_output
    
    # Otherwise, compute both base and LoRA outputs
    lora_output = self.lora_B(self.lora_A(x))
    return base_output + (lora_output * self.scaling)

PeftLinear4bit.forward = compiled_lora_linear4bit_forward

# --------------------------------------- SIMPLE FOWARD ---------------------------------------
def forward_pass(model, tokenizer, prompt="Hello, I am an AI assistant.", fullgraph=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Compile the forward pass to avoid graph breaks
    @torch.compile(fullgraph=fullgraph, dynamic=True, options=torch_compile_options)
    def forward_fn(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask)

    with torch.no_grad():
        return forward_fn(inputs.input_ids, inputs.attention_mask).logits
    
if __name__ == "__main__":
    
    max_seq_length = 1024
    torch.set_default_dtype(torch.float16)
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = "auto",
        attn_implementation = "sdpa",
        quantization_config = bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r = 32,
        lora_alpha = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
        task_type = TaskType.CAUSAL_LM,
    )

    # Get LoRA and setup model
    model = get_peft_model(model, lora_config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name: param.requires_grad_(True)
            else: param.requires_grad_(False)

    # Currently GC will cause torch.compile to be disabled, so disable it
    # model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    model.config._attn_implementation = "compiled_flex"
    
    logits = forward_pass(model, tokenizer, "Whats the capital of France?", fullgraph=False)