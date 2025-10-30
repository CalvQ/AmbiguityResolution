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
    
    def _build_clarification_prompt(self, prompt, history):
        """
        Build a prompt that asks the model to generate a clarifying question
        based on the ambiguity in the user's request.
        
        Args:
            prompt (str): The current ambiguous prompt
            history (list[str]): Previous user messages in the conversation
            
        Returns:
            str: Formatted prompt for clarification generation
        """
        guidance = (
            "You are helping a user interact with a 3D scene. The user's request is ambiguous because multiple objects match their description.\n\n"
            "Your task:\n"
            "1. Identify which objects in the scene match the user's description\n"
            "2. Find the key attributes that distinguish these objects (color, size, position, material, etc.)\n"
            "3. Ask ONE question that presents these distinguishing options\n\n"
            "Guidelines:\n"
            "- ALWAYS provide specific options in your question (e.g., 'the red one or the blue one?', 'the one on the left or by the window?')\n"
            "- Use visual properties the user can SEE (color, size, shape, material)\n"
            "- Use spatial descriptions (left/right, near X, by Y, in the corner)\n"
            "- NEVER use object IDs, numbers, or internal identifiers (don't say 'chair_1' or 'object 2')\n"
            "- Present 2-3 concrete options that narrow down the search space\n"
            "- Keep the question under 20 words\n\n"
            "Examples of GOOD questions:\n"
            "- Do you mean the red cube or the blue cube?\n"
            "- The chair by the window or the one near the door?\n"
            "- The larger sphere on the left or the smaller one on the right?\n\n"
            "Examples of BAD questions (too vague):\n"
            "- Which chair?\n"
            "- Which cup do you mean?\n"
            "- Can you be more specific?\n\n"
            "Return ONLY the clarifying question with specific options, nothing else."
        )
        
        scene_json = json.dumps(self.scene, indent=2)
        
        # Build conversation history section
        history_section = ""
        if history:
            history_section = "PREVIOUS CONVERSATION:\n"
            for i, msg in enumerate(history, 1):
                history_section += f"User {i}: {msg}\n"
            history_section += "\n"
        
        # Combine everything into one user message
        messages = [
            {
                "role": "user",
                "content": (
                    f"{guidance}\n\n"
                    "SCENE (JSON):\n```json\n"
                    f"{scene_json}\n```\n\n"
                    f"{history_section}"
                    f"CURRENT REQUEST (ambiguous):\n{prompt}\n\n"
                    "Generate a clarifying question:"
                ),
            }
        ]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            return messages[0]["content"]
        
    def _build_context_summary_prompt(self, user_response, history, clarifying_question):
        """
        Build a prompt that asks the model to create a contextualized version
        of the user's response that incorporates information from the conversation.
        
        Args:
            user_response (str): The user's latest response
            history (list[dict]): Previous conversation turns
            clarifying_question (str): The question that prompted this response
            
        Returns:
            str: Formatted prompt for context summarization
        """
        guidance = (
            "You are helping resolve ambiguity in a 3D scene interaction. "
            "The user has been answering clarifying questions to specify which object they mean.\n\n"
            "Your task: Create a SINGLE, COMPLETE statement that:\n"
            "1. Preserves the ORIGINAL ACTION/INTENT from the first user message (e.g., 'point to', 'move', 'grab', 'delete')\n"
            "2. Specifies the EXACT OBJECT using all clarifying details gathered\n\n"
            "Guidelines:\n"
            "- Keep the original action verb from the conversation start (point to, move, grab, etc.)\n"
            "- Add all relevant object attributes (color, location, size, etc.)\n"
            "- If the user said 'yes', use the option from the question they confirmed\n"
            "- If the user said 'no', exclude that option\n"
            "- If the user provided new details, incorporate them\n"
            "- Keep it concise (under 25 words)\n"
            "- Make it a complete command that could stand alone\n\n"
            "Examples:\n"
            "Initial: 'move the cube' → Q: 'Red or blue cube?' → A: 'the blue one' → Output: 'move the blue cube'\n"
            "Initial: 'point to the chair' → Q: 'Chair by window or door?' → A: 'yes' → Output: 'point to the chair by the window'\n"
            "Initial: 'delete the sphere' → Q: 'Larger or smaller?' → A: 'the larger one on the left' → Output: 'delete the larger sphere on the left'\n"
            "Initial: 'grab that' → Q: 'The red cube or blue cube?' → A: 'red' → Output: 'grab the red cube'\n\n"
            "Return ONLY the complete contextualized command, nothing else."
        )
        
        scene_json = json.dumps(self.scene, indent=2)
        
        # Build conversation history
        history_section = ""
        if history:
            history_section = "CONVERSATION SO FAR:\n"
            for i, turn in enumerate(history, 1):
                history_section += f"User: {turn['user']}\n"
                history_section += f"Assistant: {turn['assistant']}\n"
            history_section += "\n"
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"{guidance}\n\n"
                    "SCENE (JSON):\n```json\n"
                    f"{scene_json}\n```\n\n"
                    f"{history_section}"
                    f"LATEST QUESTION: {clarifying_question}\n"
                    f"USER'S RESPONSE: {user_response}\n\n"
                    "Generate the contextualized statement:"
                ),
            }
        ]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            return messages[0]["content"]
        
    def contextualize_user_response(self, user_response, history, clarifying_question):
        """
        Create a contextualized version of the user's response that includes
        all relevant information from the conversation history.
        
        Args:
            user_response (str): The user's latest response
            history (list[dict]): Previous conversation turns
            clarifying_question (str): The question that prompted this response
            
        Returns:
            str: A self-contained statement with full context
        """
        context_prompt = self._build_context_summary_prompt(
            user_response, history, clarifying_question
        )
        
        contextualized = generate(
            self.model,
            self.tokenizer,
            prompt=context_prompt,
            max_tokens=75,
            verbose=False,
        )
        
        # Clean up the response
        contextualized = contextualized.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "contextualized statement:",
            "statement:",
            "output:",
            "the user wants:",
        ]
        contextualized_lower = contextualized.lower()
        for prefix in prefixes_to_remove:
            if contextualized_lower.startswith(prefix):
                contextualized = contextualized[len(prefix):].strip()
                break
        
        print(f"\n[Contextualized]: {contextualized}")
        
        return contextualized
        
    def _generate_clarification(self, prompt, history):
        """
        Generate a clarifying question using the LLM based on the ambiguous prompt.
        
        Args:
            prompt (str): Most recent ambiguous message from the user
            history (list[str]): History of messages already sent by the user
            
        Returns:
            str: User's response to the clarification question
        """
        # Build the clarification prompt
        clarification_prompt = self._build_clarification_prompt(prompt, history)
        
        # Generate the clarifying question
        question = generate(
            self.model,
            self.tokenizer,
            prompt=clarification_prompt,
            max_tokens=75,
            verbose=False,
        )
        
        # Clean up the generated question
        question = question.strip()
        
        # Remove common prefixes the model might add
        prefixes_to_remove = [
            "clarifying question:",
            "question:",
            "here's a clarifying question:",
            "i would ask:",
        ]
        question_lower = question.lower()
        for prefix in prefixes_to_remove:
            if question_lower.startswith(prefix):
                question = question[len(prefix):].strip()
                break
        
        # Ensure it ends with a question mark
        if not question.endswith("?"):
            question += "?"
        
        # Print the clarifying question and get user's response
        # print(f"\n[Clarification needed]: {question}")
        return question
    
    def generate_clarification(self, prompt, history):
        question = self._generate_clarification(prompt, history)
        user_response = input("Your answer: ")
        final_response = self.contextualize_user_response(user_response, history, question)
        return final_response, question

    def resolve_prompt(self, prompt):
        current_prompt = prompt
        history = []
        while True:
            is_ambig, raw = self.is_prompt_ambiguous(current_prompt)
            # Print raw model output so you can "see the output"
            print(current_prompt)
            print(f"[model raw]: {raw}")

            if is_ambig is not True:
                break

            # Single-turn clarification (interactive)
            next_prompt, clarifying_question = self.generate_clarification(current_prompt, history)
            history.append({
                'user': current_prompt,
                'assistant': clarifying_question
            })
            current_prompt = next_prompt

        return current_prompt, history
    
    # for the case: not ask human to input in the command line, but have another agent as human-responsor
    # return value:
        # if no ambiguous: 'Non-Ambiguous'
        # if ambiguous: clarifying_question
    def detect_and_ask(self,prompt,history):
        if len(history) == 0:
            is_ambig, raw = self.is_prompt_ambiguous(prompt)
            if is_ambig is not True:
                return 'Non-Ambiguous'
        else:
            prompt = self.contextualize_user_response(
                prompt, history, history[-1]['assistant']
            )
        clarifying_question = self._generate_clarification(prompt, history)
        return clarifying_question
    
