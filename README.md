# Repo to Text

Repo to Text is a Python tool that converts GitHub repositories and local folders into text format to be used for either training LLMs or using copy pasting into LLM context. this repo provids AI-powered analysis of images, CSV files, and Jupyter notebooks.

## Features

- Process GitHub repositories and local folders and create a single text file for each repository
- Convert images to text by describing them using Google's Gemini Flash model (fast and free for up to 15 requests per minute)
- Covert Jupyter notebooks to text where the input (code/markdown) & output of each cell is clearly delimitered (if output contains images, they are also described using Flash)
- Describe large CSV files also using Flash instead of just dumping their entire content
- Handle API rate limits with built-in retry logic

## Installation

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/yourusername/repo_to_text.git
cd repo_to_text
pip install -e .
```

## Usage

To use Repo to Text, you need to set up your API keys first.  
The following code will create a `.env` file in your project root and load it into your environment variables:

```python
from repo_to_text.api_key_manager import initialize_api_keys

API_KEYS = {
    'GEMINI_API_KEY': 'abcdefghijklmnopqrstuvwxyz1234567890abc',
}

initialize_api_keys(api_keys_dict=API_KEYS)
```

Then, you can use the tool as follows from your Python scripts:

```python
from repo_to_text import convert_repos_to_text

# For GitHub repositories
github_repos = [
    "https://github.com/username/repo1",
    "https://github.com/username/repo2"
]
convert_repos_to_text(github_repos, "output_dir", is_github=True)

# For local folders
local_folders = [
    "/path/to/folder1",
    "/path/to/folder2"
]
convert_repos_to_text(local_folders, "output_dir", is_github=False)
```

## License

This project is licensed under the MIT License.
