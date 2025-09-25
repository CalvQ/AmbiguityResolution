import load_llm
import random

from mlx_lm.generate import generate_step, generate
from mlx_lm.sample_utils import make_sampler
import json
import mlx.core as mx
import re

class AREngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scene = None
        
    def load_model(self):
        self.model, self.tokenizer = load_llm.load_mlx_mistral_7b()
        
    def set_scene(self, scene):
        self.scene = scene
        
    def _build_prompt(self, user_prompt, scene):
        """
        Build a chat-formatted prompt that starts with a USER turn.
        We fold the 'system' guidance into the first user message to satisfy
        Mistral's template requirement (strict user/assistant alternation).
        """
        guidance = (
            "You are an expert discussing a 3D scene. "
            "Use concise spatial language, refer to object names, and reason about the ambiguity of a user's input. "
            "Return your answer in one word ONLY:\n"
            "true  -> if the input is ambiguous\n"
            "false -> if the input is not ambiguous"
        )

        scene_json = json.dumps(scene if scene is not None else self.scene, indent=2)
        # Put guidance + scene + user prompt into a single USER message
        messages = [
            {
                "role": "user",
                "content": (
                    f"{guidance}\n\n"
                    "SCENE (JSON):\n```json\n"
                    f"{scene_json}\n```\n"
                    f"{user_prompt}"
                ),
            }
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            # Fallback to raw text if the model lacks a chat_template
            return messages[0]["content"]
    
    def is_prompt_ambiguous(self, prompt, scene = None):
        prompt_str = self._build_prompt(prompt, scene)
        
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt_str,
            max_tokens=8,
            verbose=False,
        )
        
        first = re.sub(r"[\\s\\W]+$", "", str(text).strip().lower())  # strip punctuation
        if first.startswith("true"):
            return True, str(text)
        if first.startswith("false"):
            return False, str(text)

        # Fallback: if the model was chatty, search for true/false anywhere
        if "true" in first and "false" not in first:
            return True, str(text)
        if "false" in first and "true" not in first:
            return False, str(text)

        return None, str(text)
    
    def generate_clarification(self, prompt):
        print("Sorry, that dialogue was ambiguous")
    
    def resolve_prompt(self, prompt):
        current_prompt = prompt
        history = []
        while True:
            is_ambig, raw = self.is_prompt_ambiguous(current_prompt)
            # Print raw model output so you can "see the output"
            print(f"[model raw]: {raw}")

            if is_ambig is not True:
                break

            # Single-turn clarification (interactive)
            self.generate_clarification(current_prompt)
            next_prompt = input("Clarification: ")
            history.append(current_prompt)
            current_prompt = next_prompt

        return current_prompt, history