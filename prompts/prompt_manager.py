import os


class PromptManager:
    def __init__(self, base_dir="prompts"):
        self.base_dir = base_dir

    def render(self, path: str, **kwargs) -> str:
        with open(os.path.join(self.base_dir, path), encoding="utf-8") as f:
            template = f.read()
        return template.format(**kwargs)
