from langchain_core.pydantic_v1 import BaseModel, Field

class CommandGenerationRequest:
    """
    A request object that encapsulates the users prompt and other inputs. 

    TODO: 
        - Are other fields needed here for user given flags?
    """
    def __init__(self, user_prompt: str) -> None:
        self.user_prompt = user_prompt


class GeneratedCommand(BaseModel):
    """
    Describes the output from the LLM chain.
    """
    command: str = Field(description="The generated command")
    explanation: str = Field(description="A detailed explanation of the command")
    
    def __str__(self):
        return f"GeneratedCommand(command={self.command}, explanation={self.explanation})"