import json

class CommandGenerationRequest:
    def __init__(self, user_prompt: str) -> None:
        self.user_prompt = user_prompt


class GeneratedCommand:
    def __init__(self, command: str, explanation: str) -> None:
        self.command = command
        self.explanation = explanation

    @classmethod
    def from_json(cls, data):
        return cls(**data)
    
    def __str__(self):
        return f"GeneratedCommand(command={self.command}, explanation={self.explanation})"