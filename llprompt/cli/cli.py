import argparse
import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from model.command_generation_request import CommandGenerationRequest
from core.command_generator import CommandGenerator

class CLI:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="A CLI tool that uses LLMs to supercharge your shell.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self.parser.add_argument(
            'prompt',
            type=str,
            help="The prompt describing your desired outcome"
        )    

    def run(self):
        args = self.parser.parse_args()

        request = CommandGenerationRequest(user_prompt=args.prompt)

        generator = CommandGenerator().generate_bash_command(request=request)
