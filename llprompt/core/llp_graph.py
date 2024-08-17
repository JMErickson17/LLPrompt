from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_ollama import ChatOllama

from llprompt.model.command_generation_request import (
    CommandGenerationRequest,
    GeneratedCommand,
)

from .command_generator import CommandGeneratorNode
from .llp_state import LLPState

import uuid
from langsmith import traceable

run_id = str(uuid.uuid1())

# -------------------------------------------------------------------------
# MARK: LLP Configuration
# -------------------------------------------------------------------------


class LLPConfiguration(TypedDict):
    # TODO: Make these hot swappable with an env
    llm: str


# -------------------------------------------------------------------------
# MARK: LLPGraph
# -------------------------------------------------------------------------


@traceable(name="LLPGraph", run_id=run_id)
class LLPGraph:
    def __init__(self) -> None:
        self.instance_id = str(uuid.uuid1())
        graph_builder = StateGraph(LLPState, config_schema=LLPConfiguration)

        llm = ChatOllama(
            model="llama3.1",
        )

        graph_builder.add_node("generate", CommandGeneratorNode(llm=llm))

        graph_builder.add_edge(START, "generate")
        graph_builder.add_edge("generate", END)

        # TODO: Add this to context
        self.memory = SqliteSaver(
            sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        )

        self.graph = graph_builder.compile(checkpointer=self.memory)

    def generate_bash_command(
        self, request: CommandGenerationRequest
    ) -> GeneratedCommand:

        config = {"configurable": {"thread_id": self.instance_id}}

        output = self.graph.invoke(
            {"messages": [("user", request.user_prompt)]}, config=config
        )

        print("Output from graph: {}".format(output))

        return GeneratedCommand(command=output["generated_command"], explanation="")
