import load_llm
import random
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class AREngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scene = None
        

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B" #"Qwen/Qwen3-30B-Instruct-AWQ"  # 用量化版，Colab 40GB 才能跑
        )

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=256
        )

        self.model = LLM(
            model="Qwen/Qwen3-8B",
            tensor_parallel_size=1
        )

        print("[vLLM] Qwen3 loaded")

    def set_scene(self, scene):
        """
        Set the scene for ambiguity detection.
        
        Args:
            scene: Can be either:
                - dict with "environment_info" key (raw string)
                - list of structured object dictionaries
                - dict with "objects" key containing list of objects
        """
        # Handle different scene formats
        if isinstance(scene, dict):
            if "objects" in scene:
                # Structured format: {"objects": [...]}
                self.scene = scene["objects"]
            elif "environment_info" in scene:
                # Legacy format: {"environment_info": "..."}
                # Import here to avoid circular dependency
                from scene_parser import parse_environment_info
                self.scene = parse_environment_info(scene["environment_info"])
            else:
                # Unknown dict format, treat as single object
                self.scene = [scene]
        elif isinstance(scene, list):
            # Already a list of objects
            self.scene = scene
        else:
            # Fallback
            self.scene = scene
        
    def _build_prompt(self, user_prompt: str, scene: dict):
        """
        Build a chat-formatted prompt that starts with a USER turn.
        We fold the 'system' guidance into the first user message to satisfy
        Mistral's template requirement (strict user/assistant alternation).
        """
        guidance = (
            "You are an expert analyzing whether a user's request about a 3D scene is ambiguous.\n\n"
            "TASK: Determine if the user's request has ambiguity that requires clarification.\n\n"
            "STEPS:\n"
            "1. Identify what object type the user is referring to (e.g., 'sink', 'chair', 'window')\n"
            "2. Check if the user specified any attributes (color, size, position, etc.)\n"
            "3. Find all objects of that type in the scene\n"
            "4. Check if the user's description matches the actual objects\n\n"
            "TWO TYPES OF AMBIGUITY:\n\n"
            "TYPE 1 - Multiple Matches (ambiguous = true):\n"
            "- The user's request could refer to 2 or more different objects\n"
            "- Example: 'move the chair' when there are 3 chairs in the scene\n"
            "- Example: 'grab the gray trash can' when there are 2 gray trash cans\n\n"
            "TYPE 2 - Attribute Mismatch (ambiguous = true):\n"
            "- User specifies an attribute (like color) that doesn't match any objects\n"
            "- Example: 'open the yellow closet' when the only closet is tan\n"
            "- Example: 'sit on the red chair' when there are only blue chairs\n"
            "- IMPORTANT: If the user then specifies the correct attribute (e.g., 'the tan closet'), this is NO LONGER ambiguous\n"
            "- In these cases, we need to ask if they mean the actual object (with correct attributes)\n\n"
            "NOT AMBIGUOUS (false):\n"
            "- Exactly 1 object matches the description perfectly\n"
            "- Example: 'move the chair' when there is only 1 chair\n"
            "- Example: 'open the tan closet' when there is 1 tan closet\n"
            "- Example: 'grab the gray trash can' when there is 1 gray trash can\n"
            "- Example: 'sit on the darkslategray chair' when there is 1 darkslategray chair\n\n"
            "CRITICAL RULES:\n"
            "- If user specifies a color/attribute that doesn't exist → return true (need to clarify)\n"
            "- If 2+ objects match the description → return true (need to clarify which one)\n"
            "- If exactly 1 object matches perfectly → return false (no clarification needed)\n"
            "- If 0 objects match but similar objects exist → return true (attribute mismatch)\n"
            "- IMPORTANT: After clarification, if the user's description now matches an exact attribute in the scene → return false\n\n"
            "Return ONLY one word:\n"
            "true  -> if multiple matches OR attribute mismatch (needs clarification)\n"
            "false -> if exactly 1 perfect match (no clarification needed)"
        )

        # Format scene based on type
        scene_data = scene if scene is not None else self.scene
        if isinstance(scene_data, list):
            # Structured objects - format nicely
            scene_str = self._format_scene_objects(scene_data)
        elif isinstance(scene_data, dict) and "environment_info" in scene_data:
            # Legacy format
            scene_str = scene_data["environment_info"]
        else:
            # Fallback to JSON
            scene_str = json.dumps(scene_data, indent=2)
        
        # Put guidance + scene + user prompt into a single USER message
        messages = [
            {
                "role": "user",
                "content": (
                    f"{guidance}\n\n"
                    "SCENE:\n"
                    f"{scene_str}\n\n"
                    f"USER REQUEST: {user_prompt}\n\n"
                    "Is this request ambiguous? Answer with only 'true' or 'false':"
                ),
            }
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback to raw text if the model lacks a chat_template
            return messages[0]["content"]
    
    def _format_scene_objects(self, objects):
        """
        Format structured scene objects into a readable string.
        
        Args:
            objects (list[dict]): List of scene objects with type, center, size, colors
        
        Returns:
            str: Formatted scene description
        """
        if not objects:
            return "Empty scene"
        
        lines = []
        for obj in objects:
            parts = [obj.get("type", "Unknown")]
            
            # Add primary color
            colors = obj.get("colors", [])
            if colors:
                parts.append(f"({colors[0]})")
            
            # Add position
            center = obj.get("center")
            if center:
                parts.append(f"at ({center['x']:.1f}, {center['y']:.1f}, {center['z']:.1f})")
            
            # Add size descriptor if notable
            size = obj.get("size")
            if size:
                dims = [size.get("length", 0), size.get("width", 0), size.get("height", 0)]
                max_dim = max(dims) if dims else 0
                min_dim = min(dims) if dims else 0
                if max_dim > min_dim * 2:  # Significant dimension difference
                    if size.get("height", 0) > size.get("length", 0) and size.get("height", 0) > size.get("width", 0):
                        parts.append("(tall)")
                    elif size.get("length", 0) > size.get("width", 0) * 1.5 or size.get("width", 0) > size.get("length", 0) * 1.5:
                        parts.append("(long)")
            
            lines.append(" ".join(parts))
        
        return "\n".join(lines)
    
    def is_prompt_ambiguous(self, prompt, scene = None):
        prompt_str = self._build_prompt(prompt, scene)
        
        # vLLM generate
        outputs = self.model.generate(
            [prompt_str],
            self.sampling_params
        )
        text = outputs[0].outputs[0].text.strip().lower()
        
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
            "You are helping a user interact with a 3D scene. The user's request needs clarification for one of two reasons:\n"
            "1. Multiple objects match their description, OR\n"
            "2. The user described an object with incorrect attributes\n\n"
            "Your task:\n"
            "STEP 1: Determine the type of ambiguity\n"
            "- Count how many objects of the requested type exist in the scene\n"
            "- Check if the user specified attributes (color, size, etc.)\n"
            "- Verify if those attributes match the actual object(s)\n\n"
            "STEP 2: Generate appropriate clarification\n\n"
            "FOR MULTIPLE MATCHES (2+ objects of same type):\n"
            "- List all matching objects with their actual attributes\n"
            "- Ask which one they mean, using distinguishing features\n"
            "- Example: 'Do you mean the red cube or the blue cube?'\n\n"
            "FOR ATTRIBUTE MISMATCH (1 object, but user's description is wrong):\n"
            "- Identify the actual attributes of the object\n"
            "- Ask if they meant the object with its correct attributes\n"
            "- Example: User says 'yellow closet', scene has 'tan closet' → Ask: 'Did you mean the tan closet?'\n"
            "- Example: User says 'red chair', scene has 'blue chair' → Ask: 'Did you mean the blue chair?'\n\n"
            "Guidelines:\n"
            "- CRITICAL: Use ONLY attributes that appear in the scene data\n"
            "- Do NOT invent or assume attributes not listed\n"
            "- For attribute mismatch, focus on correcting the wrong attribute\n"
            "- For multiple matches, highlight distinguishing features\n"
            "- Keep questions under 20 words\n"
            "- Never use object IDs or technical references\n\n"
            "Examples of GOOD attribute mismatch questions:\n"
            "- Did you mean the tan closet? (user said 'yellow')\n"
            "- Do you mean the blue chair? (user said 'red')\n"
            "- Did you mean the large table? (user said 'small')\n\n"
            "Examples of GOOD multiple match questions:\n"
            "- Do you mean the rosybrown trash can or the darkgray trash can?\n"
            "- The chair at (2.3, 4.5) or the one at (6.1, 4.2)?\n\n"
            "Return ONLY the clarifying question, nothing else."
        )
        
        # Format scene
        if isinstance(self.scene, list):
            scene_str = self._format_scene_objects(self.scene)
        else:
            scene_str = json.dumps(self.scene, indent=2)
        
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
                    "SCENE:\n"
                    f"{scene_str}\n\n"
                    f"{history_section}"
                    f"CURRENT REQUEST (ambiguous):\n{prompt}\n\n"
                    "First, list the matching objects. Then generate a clarifying question:"
                ),
            }
        ]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
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
            "Your task: Create a statement that:\n"
            "1. Takes the COMPLETE ORIGINAL USER REQUEST (the very first message in the conversation)\n"
            "2. ONLY updates the object reference based on the clarification\n"
            "3. PRESERVES everything else from the original: structure, phrasing, tone, reasons, context\n\n"
            "CRITICAL PROCESS:\n"
            "Step 1: Find the FIRST user message in the conversation - this is the complete original request\n"
            "Step 2: Identify what needs to be clarified (the object/attribute)\n"
            "Step 3: Based on the user's response to the clarification question, determine the correct object details\n"
            "Step 4: Replace ONLY the object reference in the original sentence with the clarified version\n"
            "Step 5: Keep EVERYTHING else exactly as it was in the original\n\n"
            "WHAT TO PRESERVE (DO NOT CHANGE):\n"
            "- ✅ Polite phrases: 'Can you', 'Could you', 'Please', 'Would you mind', etc.\n"
            "- ✅ Verb forms and tenses: 'open', 'opened', 'opening', etc.\n"
            "- ✅ Reasons/purposes: 'so I can...', 'because...', 'to...', 'for...'\n"
            "- ✅ Qualifiers: 'quickly', 'carefully', 'gently', 'immediately', etc.\n"
            "- ✅ Additional context: 'when you get a chance', 'if possible', etc.\n"
            "- ✅ Question structure: If original is a question, output is a question\n"
            "- ✅ Punctuation: Keep '?', '!', or '.' as in original\n\n"
            "WHAT TO UPDATE (ONLY THIS):\n"
            "- ✅ The object reference: add/correct color, size, position as clarified\n"
            "- ✅ Wrong attributes: replace with correct ones from clarification\n\n"
            "Examples:\n"
            "Original: 'Can you move the cube please?' → Clarified: blue cube\n"
            "→ Output: 'Can you move the blue cube please?'\n"
            "NOT: 'move the blue cube' ❌\n\n"
            "Original: 'Could you please open the yellow closet so I can store my clothes?' → Clarified: tan closet\n"
            "→ Output: 'Could you please open the tan closet so I can store my clothes?'\n"
            "NOT: 'open the tan closet' ❌\n\n"
            "Original: 'I want to sit on the chair by the window' → Clarified: red chair\n"
            "→ Output: 'I want to sit on the red chair by the window'\n"
            "NOT: 'sit on the red chair' ❌\n\n"
            "Original: 'Grab that quickly!' → Clarified: red cube\n"
            "→ Output: 'Grab the red cube quickly!'\n"
            "NOT: 'grab the red cube' ❌\n\n"
            "Original: 'Delete the sphere because it's blocking the view' → Clarified: larger sphere\n"
            "→ Output: 'Delete the larger sphere because it's blocking the view'\n"
            "NOT: 'delete the larger sphere' ❌\n\n"
            "Original: 'I want to take a break and sit on the black chair.' → Clarified: darkslategray chair\n"
            "→ Output: 'I want to take a break and sit on the darkslategray chair.'\n"
            "NOT: 'sit on the darkslategray chair' ❌\n"
            "NOT: 'move the darkslategray chair' ❌\n\n"
            "Return ONLY the complete statement with full original context preserved."
        )
        
        # Format scene
        if isinstance(self.scene, list):
            scene_str = self._format_scene_objects(self.scene)
        else:
            scene_str = json.dumps(self.scene, indent=2)
        
        # FIX: Use only the ORIGINAL prompt from the first turn instead of full history
        original_prompt = history[0]['user'] if history else "(no history)"
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"{guidance}\n\n"
                    "SCENE:\n"
                    f"{scene_str}\n\n"
                    f"ORIGINAL REQUEST: {original_prompt}\n"
                    f"LATEST QUESTION: {clarifying_question}\n"
                    f"USER'S RESPONSE: {user_response}\n\n"
                    "Generate the contextualized statement:"
                ),
            }
        ]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
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
        
        outputs = self.model.generate(
            [context_prompt],
            self.sampling_params
        )
        contextualized = outputs[0].outputs[0].text.strip()
        
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
        
        print(f"[Contextualized]: {contextualized}")
        
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
        
        outputs = self.model.generate(
                    [clarification_prompt],
                    self.sampling_params
                )
        question = outputs[0].outputs[0].text.strip()
        
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
        question = question.strip("'")
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
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            is_ambig, raw = self.is_prompt_ambiguous(current_prompt)
            
            # Print raw model output so you can "see the output"
            print(f"\n[Iteration {iteration}] {current_prompt}")
            print(f"[model raw]: {raw}")

            if is_ambig is not True:
                print(f"[Resolution complete!]")
                break

            # Single-turn clarification (interactive)
            next_prompt, clarifying_question = self.generate_clarification(current_prompt, history)
            history.append({
                'user': current_prompt,
                'assistant': clarifying_question
            })
            current_prompt = next_prompt
        
        if iteration >= max_iterations:
            print(f"\n[WARNING] Max iterations ({max_iterations}) reached. Using best effort.")

        return current_prompt, history
    
    # for the case: not ask human to input in the command line, but have another agent as human-responsor
    # return value:
        # if no ambiguous: 'Non-Ambiguous'
        # if ambiguous: clarifying_question
    def detect_and_ask(self,prompt,history):
        if len(history) != 0:
            prompt = self.contextualize_user_response(
                prompt, history, history[-1]['assistant']
            ) # generate new query using contextualization version of query
        is_ambig, raw = self.is_prompt_ambiguous(prompt)
        if is_ambig is not True:
            return 'Non-Ambiguous'
        clarifying_question = self._generate_clarification(prompt, history) # ask clarification questions
        return clarifying_question
    
