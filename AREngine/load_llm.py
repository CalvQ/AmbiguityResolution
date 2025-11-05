# from mlx_lm import load
from transformers import AutoModelForCausalLM, AutoTokenizer

# def load_mlx_mistral_7b():
#     repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
#     model, tokenizer = load(repo)
#     return model, tokenizer

def load_qwen3_30b():
    """
    Load Qwen3-30B-A3B-Instruct-2507 model.
    Note: This is a 30B parameter model (3.3B activated) and requires significant GPU memory.
    Requires transformers>=4.51.0
    """
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True
    ).eval()
    
    return model, tokenizer