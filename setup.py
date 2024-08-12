from setuptools import setup, find_packages

setup(
    name="LLPrompt",
    version="0.0.1",
    description="An LLM backed CLI that generates bash commands",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",
    url="https://github.com/JMErickson17/LLPrompt",
    author="Justin Erickson",
    author_email="JMEricksonDev@gmail.com",
    license="MIT",  
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-ollama",
        "langchain-community",
        "langchain-chroma",
        "rich",
        "typer",
        "langsmith",
        "beautifulsoup4"
    ],
    python_requires='>=3.12',
    entry_points={
        "console_scripts": [
            "llp=llprompt.main.py",
        ],
    }
)
