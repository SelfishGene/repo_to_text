import os
import dotenv

# API Key Configurations
API_KEYS = {
    'GEMINI_API_KEY': 'abcdefghijklmnopqrstuvwxyz1234567890abc',
}

def save_env_file(api_keys_dict=API_KEYS):
    """Create a .env file with API keys if it doesn't exist."""
    env_file = '.env'
    if not os.path.exists(env_file):
        print("Creating .env file...")
        with open(env_file, 'w') as f:
            for key, value in api_keys_dict.items():
                f.write(f"{key}={value}\n")
        print(f".env file created at {os.path.abspath(env_file)}")
        return False
    return True

def setup_api_keys():
    """Load API keys from .env file and set them as environment variables."""
    # Load the .env file
    dotenv.load_dotenv()
    
    # Check if all required keys are present
    missing_keys = [key for key in API_KEYS.keys() if not os.getenv(key)]
    
    if missing_keys:
        print(f"Error: The following API keys are missing in the .env file: {', '.join(missing_keys)}")
        print("Please add them to the .env file and run the script again.")
        return False
    else:
        print("API keys loaded successfully.")
        # Set environment variables
        for key in API_KEYS.keys():
            os.environ[key] = os.getenv(key)
        return True

def initialize_api_keys(api_keys_dict=None):

    # if we are given a dictionary of API keys, override the .env file
    if api_keys_dict is not None:
        env_file_exists = save_env_file(api_keys_dict)

    # check if .env file exists
    env_file_exists = os.path.exists('.env')

    if env_file_exists:
        return setup_api_keys()
    else:
        if save_env_file():
            return setup_api_keys()
    
    return False

def print_api_keys():
    """Print all loaded API keys to the screen."""
    for key in API_KEYS.keys():
        print(f'{key} = {os.environ.get(key, "Not set")}')

if __name__ == "__main__":
    if initialize_api_keys():
        print_api_keys()
    else:
        print("Failed to initialize API keys. Please check your .env file and try again.")
