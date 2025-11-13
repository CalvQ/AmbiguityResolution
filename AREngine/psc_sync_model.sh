SRC="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507"
DEST="$LOCAL/models/Qwen3-30B-A3B-Instruct-2507"
mkdir -p "$DEST"

rsync -a --info=progress2 "$SRC"/ "$DEST"/

export HF_HOME="$LOCAL/hf_home"
export HUGGINGFACE_HUB_CACHE="$LOCAL/hf_cache"
export TRANSFORMERS_CACHE="$LOCAL/hf_cache"
export TRANSFORMERS_OFFLINE=1

