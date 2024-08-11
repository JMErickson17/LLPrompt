from model.command_generation_request import CommandGenerationRequest
from model.command_generation_request import GeneratedCommand

from langchain_ollama import ChatOllama
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langsmith import traceable

import uuid

class CommandGenerator:
    """
    Instances of the CommandGenerator are responsible for building 
    and invoking the chain that generates the bash command.

    TODO:
        1. Add ability to use different models.
    """

    ##
    ## Init
    ##

    def __init__(self) -> None:
        self.instance_id = str(uuid.uuid1())
        self.chat_history = ChatMessageHistory()

        self.llm = ChatOllama(
            model='llama3.1',
        )

        self.json_parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            ('system', CommandGenerator.system_prompt()),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}'),
        ])

        chain = prompt | self.llm

        self.chat = RunnableWithMessageHistory(
            chain, 
            lambda session_id: self.chat_history,
            input_messages_key='input',
            history_messages_key='history'
        )    
    
    ##
    ## Chain Invocation
    ##

    @traceable
    def generate_bash_command(self, request: CommandGenerationRequest, is_revision: bool) -> GeneratedCommand:
        result = self.chat.invoke(
            {'input': CommandGenerator.formatted_user_prompt(request.user_prompt, is_revision=is_revision)},
            config={"configurable": {"session_id": self.instance_id}},
        )

        return GeneratedCommand.from_json(
            self.json_parser.parse(
                result.content
            )
        )
    
    ##
    ## Prompts
    ##

    @classmethod
    def system_prompt(cls) -> str:
        return """
        You are a highly skilled Linux systems administrator. You can generate efficient and correct bash commands for any task. 

        The current operating system is MacOS.

        Write the bash command(s) that will accomplish the task described. Also provide a short explanation of what the command will do.

        Your response should be structured as a json object that contains two keys: 'command' and 'explanation'. Only the json object should be returned.
        """
    
    @classmethod
    def formatted_user_prompt(cls, user_prompt: str, is_revision) -> str:
        if not is_revision:
            return f"""
            Task: {user_prompt}
            """
        else:
            return f"""
            Please revise the previous bash command using the following revision instructions.

            {user_prompt}
        """