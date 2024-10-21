#%% Imports

import os
import time
from repo_to_text_core import convert_repos_to_text
from api_key_manager import initialize_api_keys, print_api_keys

#%% Main
if __name__ == "__main__":

    #%% Setup API keys

    API_KEYS = {
        'GEMINI_API_KEY': 'abcdefghijklmnopqrstuvwxyz1234567890abc',
    }

    initialize_api_keys(api_keys_dict=API_KEYS)
    print_api_keys()

    #%% Example usage for GitHub repositories

    github_repos_urls_list = [
        "https://github.com/SelfishGene/SFHQ-dataset",
        "https://github.com/SelfishGene/SFHQ-T2I-dataset",
        "https://github.com/SelfishGene/ImSME-dataset",
        "https://github.com/SelfishGene/neuron_as_deep_net",
        "https://github.com/SelfishGene/my_kaggle_notebooks",
        "https://github.com/SelfishGene/filter_and_fire_neuron",
        "https://github.com/SelfishGene/a_chatgpt_never_forgets",
        "https://github.com/SelfishGene/visual_taste_approximator",
        "https://github.com/anthropics/anthropic-quickstarts",
        "https://github.com/anthropics/anthropic-cookbook",
        "https://github.com/openai/openai-cookbook",
        "https://github.com/openai/openai-python",
    ]
    github_output_dir = r"data/repos_as_text_github_urls"
    
    start_time = time.time()
    print("Processing GitHub repositories...")
    print('---------------------------------')
    convert_repos_to_text(github_repos_urls_list, github_output_dir, is_github=True, 
                          describe_images=True, describe_csv_files=True, process_notebooks=True)

    duration_min = (time.time() - start_time) / 60
    print('---------------------------------------------------------')
    print(f"Finished processing {len(github_repos_urls_list)} GitHub repositories! Took {duration_min:.2f} minutes")
    print('---------------------------------------------------------')

    #%% Example usage for local folders

    local_folder_paths_list = [
        r"code/SFHQ-dataset",
        r"code/SFHQ-T2I-dataset",
        r"code/ImSME-dataset",
        r"code/neuron_as_deep_net",
        r"code/my_kaggle_notebooks",
        r"code/filter_and_fire_neuron",
        r"code/a_chatgpt_never_forgets",
        r"code/visual_taste_approximator",
    ]
    folders_output_dir = r"data/repos_as_text_local_folders"
    
    start_time = time.time()
    print("\nProcessing local folders...")
    print('-----------------------------')
    convert_repos_to_text(local_folder_paths_list, folders_output_dir, is_github=False, 
                          describe_images=True, describe_csv_files=True, process_notebooks=True)

    duration_min = (time.time() - start_time) / 60
    print('---------------------------------------------------------')
    print(f"Finished processing {len(local_folder_paths_list)} local folders! Took {duration_min:.2f} minutes")
    print('---------------------------------------------------------')

    #%% display the first 10000 chars of the file in the output directory of local folders

    filename_to_show = 'SFHQ-T2I-dataset.txt'
    file_path = os.path.join(folders_output_dir, filename_to_show)

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    print(content[:10000])

#%%