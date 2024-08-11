import typer
import os
import sys

from rich.console import Console
from rich.table import Table

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from model.command_generation_request import CommandGenerationRequest, GeneratedCommand
from model.user_input_options import UserInputOption
from core.command_generator import CommandGenerator

console = Console(width=100)
llp = typer.Typer()

command_generator = CommandGenerator()

@llp.command()
def generate_shell_command(prompt: str):

    user_accepted_generated_command = False
    
    generated_command = generate_command(prompt, is_initial=True)

    display_table(generated_command)
    
    while not user_accepted_generated_command:

        user_input = typer.prompt('Enter an option to proceed')

        try:
            selected_option = list(UserInputOption)[int(user_input) - 1]

            if selected_option == UserInputOption.REVISE_PROMPT:
                revision_prompt = typer.prompt('Describe the revisions you would like')
                generated_command = generate_command(revision_prompt, is_initial=False)
                display_table(generated_command)

            else:
                #TODO: Handle the other cases
                user_accepted_generated_command = True
                break

        except (IndexError, ValueError):
            console.print('Invalid selection')

def generate_command(prompt: str, is_initial: bool) -> GeneratedCommand:
     with console.status('Generating command'):

        request = CommandGenerationRequest(
            user_prompt=prompt
        )

        return command_generator.generate_bash_command(
            request=request,
            is_revision=not is_initial
        )
     
def display_table(generated_command: GeneratedCommand):
    console.print()

    console.print(
        generated_command.command, 
        style='green', 
        justify='center', 
        end='\n\n'
    )

    console.print(
        generated_command.explanation,
        end='\n\n'
    )

    console.print(
        'Warning: Commands generated by LLPrompt can cause irreversible damage to your computer. Please inspect the command thoroughly before running it.', 
        style='red bold'
    )

    table = Table(title='Options')
    table.add_column('Index')
    table.add_column('Option')

    for index, option in enumerate(UserInputOption, start=1):
        table.add_row(str(index), option.name.capitalize())

    console.print(table)

if __name__ == "__main__":
    llp()