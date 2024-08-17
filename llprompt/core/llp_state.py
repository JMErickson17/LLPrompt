from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

class LLPState(TypedDict):
    messages: Annotated[list, add_messages]
    generated_command: Optional[str]