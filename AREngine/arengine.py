import load_llm
import random

class AREngine:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        pass
    
    def is_prompt_ambiguous(self, prompt):
        return random.random() < 0.5
    
    def generate_clarification(self, prompt):
        print("Sorry, that dialogue was ambiguous")
    
    def resolve_prompt(self, prompt):
        current_prompt = prompt
        history = []
        while self.is_prompt_ambiguous(current_prompt):
            # Engage in a single turn dialogue here
            self.generate_clarification(current_prompt)
            next_prompt = input("Clarification:")
            history.append(current_prompt)
            current_prompt = next_prompt
        return current_prompt, history