from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import random

class Prompter:
    # Task: Write prompts & shot_examples
    PROMPT_TEMPLATES = {
        "basic": # TODO: basic prompt,
        "instruction": # TODO: instruction prompt,
        "few_shot": # TODO: few-shot prompt,
        "structured": "Here is a definition and an example for the word.\nword: {word}\ndefinition: "
    }

    def __init__(self, template_type="structured"):
        self.template_type = template_type
        self.template = self.PROMPT_TEMPLATES[template_type]
        
    def build_prompt(self, word, examples=None, n_shots=3):
        if self.template_type == "few_shot" and examples:
            shot_examples = "\n".join(
                f"word: {ex['word']}\ndefinition: {ex['definition']}\nexample: {ex['example']}"
                for ex in random.sample(examples, min(n_shots, len(examples)))
            )
            return self.template.format(word=word, examples=shot_examples)
        return self.template.format(word=word)
