class UserPromptFormatter:
    
    @classmethod
    def system_prompt(cls) -> str:
        return """You are a highly skilled Linux systems administrator. You can generate efficient and correct bash commands for any task. 

        The current operating system is MacOS.

        Write the bash command(s) that will accomplish the task described. 

        Your response should only contain the command and nothing else. 
        For example, if asked to generate a command that lists all files, you would return 'ls'.
        """

    @classmethod
    def formatted_user_prompt(cls, user_prompt: str, is_revision) -> str:
        if not is_revision:
            return f"Task: {user_prompt}"
        else:
            return f"""Please revise the previous bash command using the following revision instructions.

            {user_prompt}
            """

