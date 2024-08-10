from model.command_generation_request import CommandGenerationRequest
from model.command_generation_request import GeneratedCommand

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage

class CommandGenerator:
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model='llama3.1',
            verbose=True
        )

        self.json_parser = JsonOutputParser()

        initial_prompt = ChatPromptTemplate.from_messages([
            ('system', CommandGenerator.system_prompt()),
            MessagesPlaceholder(variable_name='history'),
            ('human', CommandGenerator.initial_prompt()),
        ])

        self.chat_runnable = self.create_chat_runnable(
            initial_prompt
        )

    def create_chat_runnable(self, prompt_template: ChatPromptTemplate) -> RunnableWithMessageHistory:
        self.runnable = prompt_template | self.llm | self.json_parser

        return RunnableWithMessageHistory(
            self.runnable, 
            self.get_session_history,
            input_messages_key='task_description',
            history_messages_key='history'
        )

    def get_session_history(self, session_id: int):
        return SQLChatMessageHistory(session_id, connection='sqlite:///chat_history.db')
    
    ##
    ## Invocation
    ##

    def generate_bash_command(self, request: CommandGenerationRequest) -> GeneratedCommand:
        result = self.chat_runnable.invoke(
            {'task_description': request.user_prompt},
            config={"configurable": {"session_id": "1"}},
        )

        return GeneratedCommand.from_json(result)
    
    def revise_bash_command(self, request: CommandGenerationRequest) -> GeneratedCommand:
        revision_prompt = ChatPromptTemplate.from_messages([
            ('system', CommandGenerator.system_prompt()),
            MessagesPlaceholder(variable_name='history'),
            ('human', CommandGenerator.revision_prompt()),
        ])

        self.chat_runnable = self.create_chat_runnable(
            revision_prompt
        )

        return self.generate_bash_command(request=request)
    
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
    def initial_prompt(cls) -> str:
        return """
        Task: {task_description}
        """
    
    @classmethod
    def revision_prompt(cls) -> str:
        return """
        Please revise the previous bash command using the following revision instructions.

        {task_description}
        """
