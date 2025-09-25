from mlx_lm import load

def load_mlx_mistral_7b():
    repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    model, tokenizer = load(repo)
    return model, tokenizer