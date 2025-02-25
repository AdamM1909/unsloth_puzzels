from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

torch.set_default_dtype(torch.float16)
torch_compile_options = torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : True,
    "triton.cudagraphs" : False,
}

# bnb_config = BitsAndBytesConfig(
#         load_in_4bit              = True,
#         bnb_4bit_use_double_quant = True,
#         bnb_4bit_quant_type       = "nf4",
#         bnb_4bit_compute_dtype    = torch.float16,
#     )

lora_config = LoraConfig(
        r = 32,
        lora_alpha = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
        task_type = TaskType.CAUSAL_LM,
    )

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_llama_mlp(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

import transformers.models.llama.modeling_llama
transformers.models.llama.modeling_llama.LlamaMLP.forward = compiled_llama_mlp




if __name__ == "__main__":
    
    max_seq_length = 128
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    dataset = Dataset.from_dict({"text": ["This is a simple test sentence." for _ in range(10)]})
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = "auto",
        attn_implementation = "sdpa",
        # quantization_config = bnb_config,
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
    
    trainer = SFTTrainer(
            model = model,
            train_dataset = dataset,
            processing_class = tokenizer,
            args = SFTConfig(
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 2,
                warmup_steps = 1,
                max_steps = 1,
                logging_steps = 1,
                output_dir = "outputs",
                seed = 3407,
                max_seq_length = max_seq_length,
                fp16 = model.get_input_embeddings().weight.dtype == torch.float16,
                bf16 = model.get_input_embeddings().weight.dtype == torch.bfloat16,
                report_to = "none", # For W&B
                dataset_num_proc = 4,
            )
    )
    trainer.train()