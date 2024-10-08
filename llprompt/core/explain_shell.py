import json
from typing import Annotated, List, Optional, TypedDict
import bs4
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

from model.command_generation_request import GeneratedCommand

from langchain_core.runnables import Runnable, RunnableConfig

import urllib.parse

from langsmith import traceable


class ExplainShell(Runnable):
    """
    ExplainShell is an experimental Runnable that uses a retriever to fetch 
    the contents of the of the explainshell.com web page to describe the generated
    command in detail and with high accuracy.

    TODO: 
    1. Improve error handling with included retries. 
    2. Is this the best architecture? Should there be an orchestrator layer that chains this and the command_generator?
    3. Can these system prompts be improved?
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    @traceable
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> GeneratedCommand:
        prompt = ChatPromptTemplate.from_messages(
            [("system", ExplainShell.system_prompt()), ("human", "{input}")]
        )

        explain_shell_chain = (
            {   
                "input": lambda dict: dict["input"],
                "context": lambda dict: dict["context"],
            }
            | prompt
            | self.llm.with_structured_output(GeneratedCommand)
        )

        retriever = self.create_retriever(input)
        retrieve_explain_shell = (lambda x: x["input"]) | retriever

        chain = RunnablePassthrough.assign(
            context=retrieve_explain_shell
        ).assign(
            generated_command=explain_shell_chain
        )

        result = chain.invoke(
            {"input": ExplainShell.explain_shell_prompt(input)}
        ).get('generated_command')

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
