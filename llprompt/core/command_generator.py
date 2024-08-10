from model.command_generation_request import CommandGenerationRequest

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class CommandGenerator:
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model='llama3.1',
            verbose=True
        )

    def generate_bash_command(self, request: CommandGenerationRequest):
        prompt = self.prompt_template().format(task_description=request.user_prompt)
        model_response = self.llm.invoke(prompt)
        print('Unparsed Output: {}'.format(model_response))

        parsed_response = JsonOutputParser().parse(model_response.content)
        print('Parsed Output: {}'.format(parsed_response))

    def prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
            You are a highly skilled Linux systems administrator. You can generate efficient and correct bash commands for any task. 

            Task: {task_description}

            Write the bash command(s) that will accomplish the task described above. Also provide a short explanation of what the command will do.

            Your response should be structured as a json object that contains two keys: 'command' and 'explanation'. Only the json object should be returned.                              
            """,
            input_variables=['task_description']
        )
