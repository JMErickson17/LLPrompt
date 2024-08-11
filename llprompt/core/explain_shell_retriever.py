import json
from typing import Annotated, List, Optional, TypedDict
import bs4
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

from langchain_core.runnables import Runnable, RunnableConfig

import urllib.parse

from langsmith import traceable

class ExplainShellExplanation(TypedDict):
    """An explanation of the shell command, with sources."""

    # command: str
    answer: str
    sources: Annotated[
        List[str],
        ...,
        "List of sources used to answer the question",
    ]


class ExplainShellRetriever(Runnable):
    def __init__(self, llm) -> None:
        self.llm = llm

    @traceable
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> ExplainShellExplanation:
        prompt = ChatPromptTemplate.from_messages(
            [("system", ExplainShellRetriever.system_prompt()), ("human", "{input}")]
        )

        explain_shell_chain = (
            {   
                "input": lambda dict: dict["input"],
                "context": lambda dict: dict["context"],
            }
            | prompt
            | self.llm.with_structured_output(ExplainShellExplanation)
        )

        retriever = self.create_retriever(input)
        retrieve_explain_shell = (lambda x: x["input"]) | retriever

        chain = RunnablePassthrough.assign(
            context=retrieve_explain_shell
        ).assign(
            answer=explain_shell_chain
        )

        result = chain.invoke(
            {"input": ExplainShellRetriever.explain_shell_prompt(input)}
        )

        return result

    @traceable
    def create_retriever(self, command: str):
        loader = WebBaseLoader(
            web_path="https://explainshell.com/explain?cmd={}".format(
                urllib.parse.quote_plus(command.strip())
            ),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    id='help'
                )
            )
        )

        docs = loader.load()

        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=OllamaEmbeddings(model="llama3.1")
        )

        return vector_store.as_retriever()

    ##
    ## Prompts
    ##

    # Using the data source provided, return a json object that contains a 'command' field with the original command, and an 'explanation' field that contains a markdown string explaining the shell command in detail. 

    @classmethod
    def system_prompt(cls) -> str:
        return """
        You are a highly skilled Linux systems administrator tasked with describing what a given bash command will do in detail. 

        Use only the following information to describe the command:

        Your response should be highly detailed and describe each flag and variable. 

        {context}
        """

    @classmethod
    def explain_shell_prompt(cls, command: str) -> str:
        return f"""
        Using the provided context, describe the following command:

        ${command}
        """
