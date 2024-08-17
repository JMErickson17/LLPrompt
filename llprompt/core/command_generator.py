from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable

from .user_prompt_formatter import UserPromptFormatter
from .llp_state import LLPState


@traceable(name="LLPGraph")
class CommandGeneratorNode:
    """
    Instances of the CommandGenerator are responsible for building
    and invoking the chain that generates the bash command.

    TODO:
        1. Add ability to use different models.
        2. Add error handling/recovery for model output, with retries (Agent?)
        3. Add ability to test generated command for accuracy?
    """

    def __init__(self, llm) -> None:
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", UserPromptFormatter.system_prompt()),
                ("placeholder", "{messages}"),
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def __call__(self, state: LLPState):
        is_revision = state["generated_command"] is not None

        latest_user_prompt = state["messages"][-1].content
        state["messages"][-1].content = UserPromptFormatter.formatted_user_prompt(
            latest_user_prompt, is_revision=is_revision
        )

        messages = state["messages"]

        output = self.chain.invoke(
            {
                "messages": messages,
            }
        )

        return {"messages": messages, "generated_command": output}
