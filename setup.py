from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="repo_to_text",
    version="0.1.0",
    author="David Beniaguev",
    description="A tool to convert repositories and local folders to text format for LLM copy pasting and training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SelfishGene/repo_to_text",
    python_requires=">=3.7",
    install_requires=[
        "requests",
        "Pillow",
        "pandas",
        "google-generativeai",
        "python-dotenv",
    ],
)
