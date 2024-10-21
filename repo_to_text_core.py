#%% Imports

import os
import io
import csv
import time
import json
import shutil
import base64
import random
import zipfile
import fnmatch
import requests
import mimetypes
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from google.api_core import exceptions

#%% Gemini Defenitions and Functions

GEMINI_MODEL_NAME = "gemini-1.5-flash-002"

def is_image_file(file_path):
    return Path(file_path).suffix.lower() in {'.jpg', '.jpeg', '.png'}

def is_csv_file(file_path):
    return Path(file_path).suffix.lower() == '.csv'

def is_ipython_notebook(file_path):
    return Path(file_path).suffix.lower() == '.ipynb'

def setup_gemini_model():
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    gemini_model = genai.GenerativeModel(model_name=f"models/{GEMINI_MODEL_NAME}")
    return gemini_model

def analyze_image(image_path, model, temperature=0.7):

    image = Image.open(image_path)
    prompt = "Describe the structure and contents of this image in a few paragraphs"
    
    input_content = [image, prompt]
    response = model.generate_content(
        input_content,
        generation_config={"temperature": temperature}
    )
    
    # output_token_count = response.usage_metadata.candidates_token_count

    current_date = datetime.now().strftime("%Y-%m-%d")
    description = f"Description of the following image by {GEMINI_MODEL_NAME}, performed on {current_date}:\n{response.text}"
    
    return description

def process_csv_file(file_path, model):
    """Process a CSV file and return its description."""
    df = pd.read_csv(file_path)
    row_count = len(df)
    col_count = len(df.columns)
    col_names = df.columns[:10].tolist()

    programmatic_description = (
        f"CSV file with {row_count} rows and {col_count} columns. "
        f"First several column names are {col_names}"
    )

    if row_count > 50:
        # For larger CSV files, use Gemini to generate a description
        csv_content = df.head(50).to_string(index=False)
        prompt = f"Describe the following CSV file in a few paragraphs in abstract terms. Focus on the structure, potential purpose, and any patterns you notice:\n\n{csv_content}"
        
        response = model.generate_content(prompt)
        gemini_description = response.text
    else:
        # For smaller CSV files, use the entire content
        gemini_description = df.to_string(index=False)

    return f"{programmatic_description}\n\n{gemini_description}"

def analyze_image_with_retries(image_path, model, temperature=0.7, max_retries=10, initial_sleep=6, show_prints=False):
    start_time = time.time()
    retries = 0
    sleep_time = initial_sleep

    while retries < max_retries:
        try:
            image = Image.open(image_path)
            prompt = "Describe the structure and contents of this image in a few paragraphs"
            
            input_content = [image, prompt]
            response = model.generate_content(
                input_content,
                generation_config={"temperature": temperature}
            )
            time.sleep(random.uniform(0.2, 0.5))

            current_date = datetime.now().strftime("%Y-%m-%d")
            try:
                description = f"Description of the following image by {model.model_name}, performed on {current_date}:\n{response.text}"
            except:
                description = f"failed to describe image by {model.model_name}"
            
            end_time = time.time()
            if show_prints:
                print(f"Image analysis completed in {end_time - start_time:.2f} seconds after {retries} retries.")
            
            return description

        except exceptions.GoogleAPICallError as e:
            retries += 1
            if show_prints:
                print(f"Attempt {retries} failed. Retrying in ~{sleep_time} seconds...")
            time.sleep(random.uniform(sleep_time - 1, sleep_time + 1))
            sleep_time = min(sleep_time * 1.3, 60)

    raise Exception(f"Failed to analyze image after {max_retries} attempts.")

def process_csv_file_with_retries(file_path, model, max_retries=10, initial_sleep=6, show_prints=False):
    start_time = time.time()
    retries = 0
    sleep_time = initial_sleep

    while retries < max_retries:
        try:
            df = pd.read_csv(file_path)
            row_count = len(df)
            col_count = len(df.columns)
            col_names = df.columns[:10].tolist()

            programmatic_description = (
                f"CSV file with {row_count} rows and {col_count} columns. "
                f"First several column names are {col_names}"
            )

            if row_count > 50:
                # For larger CSV files, use Gemini to generate a description
                csv_content = df.head(50).to_string(index=False)
                prompt = f"Describe the following CSV file in a few paragraphs in abstract terms. Focus on the structure, potential purpose, and any patterns you notice:\n\n{csv_content}"
                
                response = model.generate_content(prompt)
                try:
                    gemini_description = response.text
                except:
                    gemini_description = f"failed to describe CSV file by {model.model_name}"
                time.sleep(random.uniform(0.2, 0.5))
            else:
                # For smaller CSV files, use the entire content
                gemini_description = df.to_string(index=False)

            end_time = time.time()
            if show_prints:
                print(f"CSV processing completed in {end_time - start_time:.2f} seconds after {retries} retries.")

            return f"{programmatic_description}\n\n{gemini_description}"

        except exceptions.GoogleAPICallError as e:
            retries += 1
            if show_prints:
                print(f"Attempt {retries} failed. Retrying in ~{sleep_time} seconds...")
            time.sleep(random.uniform(sleep_time - 1, sleep_time + 1))
            sleep_time = min(sleep_time * 1.3, 60)

    raise Exception(f"Failed to process CSV file after {max_retries} attempts.")

def get_image_from_data(image_data):
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    return image

def convert_ipython_to_cell_list(notebook_filename):
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb.get('cells', [])
    cell_list = []
    for idx, cell in enumerate(cells):
        ipython_cell = {}
        cell_type = cell.get('cell_type', '')
        source = ''.join(cell.get('source', ''))
        ipython_cell['input text'] = source
        if cell_type == 'code':
            if source.lstrip().startswith('!'):
                ipython_cell['input cell type'] = 'shell'
            else:
                ipython_cell['input cell type'] = 'python'
            output_text = ''
            output_figures = []
            outputs = cell.get('outputs', [])
            for output in outputs:
                output_type = output.get('output_type', '')
                if output_type == 'stream':
                    text = ''.join(output.get('text', ''))
                    output_text += text
                elif output_type in ('execute_result', 'display_data'):
                    data = output.get('data', {})
                    if 'text/plain' in data:
                        text = ''.join(data['text/plain'])
                        output_text += text + '\n'
                    if 'image/png' in data:
                        image_data = data['image/png']
                        image = get_image_from_data(image_data)
                        output_figures.append(image)
                elif output_type == 'error':
                    traceback = output.get('traceback', [])
                    text = ''.join(traceback)
                    output_text += text + '\n'
            ipython_cell['output_text'] = output_text
            ipython_cell['output_figures'] = output_figures
        elif cell_type == 'markdown':
            ipython_cell['input cell type'] = 'markdown'
            ipython_cell['output_text'] = ''
            ipython_cell['output_figures'] = []
        else:
            ipython_cell['input cell type'] = cell_type
            ipython_cell['output_text'] = ''
            ipython_cell['output_figures'] = []
        ipython_cell['cell metadata'] = cell.get('metadata', {})
        cell_list.append(ipython_cell)
    return cell_list

def process_ipython_notebook_file(notebook_filename, gemini_model, temp_dir='temp_repo_folder'):
    cell_list = convert_ipython_to_cell_list(notebook_filename)
    
    cell_delimiter_length = 60
    output_text = "-" * cell_delimiter_length + "\n"
    images_processed = 0

    for idx, cell in enumerate(cell_list):
        output_text += f"Cell index: {idx + 1}\n"
        output_text += f"Input Cell Type: {cell['input cell type']}\n"
        output_text += "Input Text:\n"
        output_text += "-----------\n"
        output_text += cell['input text']
        output_text += "\n\n"
        output_text += "Output Text:\n"
        output_text += "------------\n"
        output_text += cell['output_text'] + "\n"
        
        # Process images if present
        images = cell['output_figures']
        if len(images) > 0:
            output_text += "\n"
            output_text += "Output Images:\n"
            output_text += "--------------\n"
        for img_idx, img in enumerate(images):
            # Generate a unique filename for the temporary image
            temp_image_path = os.path.join(temp_dir, f"temp_image_{idx}_{img_idx}.png")
            
            # Save the PIL Image object as a temporary file
            img.save(temp_image_path, format='PNG')
            
            # Analyze the image using the existing analyze_image function
            image_description = analyze_image_with_retries(temp_image_path, gemini_model)
            output_text += f"\nImage {img_idx + 1} Description:\n{image_description}\n"
            images_processed += 1
        
        output_text += "\n"
        output_text += "-" * cell_delimiter_length + "\n"
    
    print(f"Total images processed in this notebook: {images_processed}")
    return output_text, images_processed

#%% Functions

def should_ignore(path):
    """Check if a file or folder should be ignored, including parent directories."""
    path = Path(path)
    
    exact_ignores = {'.git', '.github', '.venv', 'node_modules', '.next', '__pycache__', '.DS_Store'}
    pattern_ignores = ['*.pyc', '*.log', '.env*', '.venv*', 'package-lock.json', 'yarn.lock']
    ignored_extensions = {'.exe', '.dll', '.so', '.dylib'}
    max_file_size_MB = 10

    # Check all parent directories
    for parent in path.parents:
        if parent.name in exact_ignores or any(fnmatch.fnmatch(parent.name, pattern) for pattern in pattern_ignores):
            return True

    name = path.name
    
    if name in exact_ignores:
        return True
    
    for pattern in pattern_ignores:
        if fnmatch.fnmatch(name, pattern):
            return True
    
    if name.startswith('.') and name != '.':
        return True
    
    if path.is_file():
        if path.stat().st_size > max_file_size_MB * 1024 * 1024:
            return True
        
        if path.suffix.lower() in ignored_extensions:
            return True
    
    return False

def is_text_file(file_path):
    """Check if a file is a text file."""
    text_file_extensions = {
        '.json', '.txt', '.py', '.js', '.html', '.css', '.md', 
        '.hoc', '.mod', '.asc', '.c', '.h', '.cpp',
    }

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('text'):
        return True
    
    return Path(file_path).suffix.lower() in text_file_extensions

def generate_file_structure(root_dir):
    """Generate a file structure representation, including folder names, while respecting ignore rules."""
    file_structure = []
    for root, dirs, files in os.walk(root_dir):
        # Apply should_ignore to directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
        
        level = root.replace(root_dir, '').count(os.sep)
        indent = '    ' * level
        folder_name = os.path.basename(root)
        
        if folder_name and root != root_dir:  # Skip the root directory itself
            if not should_ignore(root):
                file_structure.append(f"{indent}{folder_name}/")
            else:
                continue  # Skip this directory and its contents
        
        subindent = '    ' * (level + 1)
        for f in files:
            file_path = os.path.join(root, f)
            if not should_ignore(file_path):
                file_structure.append(f"{subindent}{f}")
    
    return file_structure

def read_file_contents(file_path):
    """Read and return the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

def write_to_output(file_path, content, mode='a'):
    """Write content to the output file."""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(content + '\n')

def delimiter_block(content):
    """Create a delimiter block for content."""
    delimiter = '=' * 80
    return f"{delimiter}\n{content}\n{delimiter}"

def file_content_block(file_path, content):
    """Create a content block for a file."""
    delimiter = '=' * 80
    path_delimiter = '=' * (len(str(file_path)) + 1)
    return f"{delimiter}\n{file_path}:\n{path_delimiter}\n{content}\n{delimiter}"

def download_repo_as_zip(repo_url, download_dir):
    """Download a GitHub repository as a zip file and extract it."""
    repo_name = repo_url.split('/')[-1]
    branches = ['main', 'master']
    for branch in branches:
        zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
        print(f"Trying to download from {zip_url}")
        response = requests.get(zip_url)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            zip_path = os.path.join(download_dir, f"{repo_name}.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(zip_path)
            extracted_dir = os.path.join(download_dir, f"{repo_name}-{branch}")
            return repo_name, extracted_dir
    raise Exception(f"Failed to download repository from both main and master branches: {repo_url}")

#%% Core Functions

def process_github_repo(repo_url, output_dir, download_dir='temp_repo_folder', describe_images=True, describe_csv_files=True, process_notebooks=True):
    """Process a GitHub repository and create a dataset."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    repo_name, extracted_dir = download_repo_as_zip(repo_url, download_dir)
    output_file = os.path.join(output_dir, f"{repo_name}.txt")

    # Write repo title and link
    date_string = datetime.now().strftime("%Y-%m-%d")
    starting_string = f"repo title: {repo_name}\nrepo link: {repo_url}\ndate processed: {date_string}"
    write_to_output(output_file, delimiter_block(starting_string), mode='w')

    # Write file structure
    file_structure = generate_file_structure(extracted_dir)
    file_structure_content = '\n'.join(file_structure)
    write_to_output(output_file, file_content_block("repo file structure", file_structure_content))

    # Set up Gemini model if needed
    gemini_model = setup_gemini_model() if (describe_images or describe_csv_files or process_notebooks) else None

    # Process all files
    all_files = []
    images_described = 0
    csvs_processed = 0
    notebooks_processed = 0
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not should_ignore(file_path) and (is_text_file(file_path) or 
                                                 (describe_images and is_image_file(file_path)) or 
                                                 (describe_csv_files and is_csv_file(file_path)) or
                                                 (process_notebooks and is_ipython_notebook(file_path))):
                all_files.append(file_path)

    # Find and process README file
    readme_files = [f for f in all_files if os.path.basename(f).lower().startswith('readme')]
    if readme_files:
        readme_file = readme_files[0]
        readme_content = read_file_contents(readme_file)
        write_to_output(output_file, file_content_block(os.path.relpath(readme_file, extracted_dir), readme_content))
        all_files.remove(readme_file)  # Remove README from all_files to avoid duplication

    # Process remaining files
    random.shuffle(all_files)
    for file_path in all_files:
        if is_text_file(file_path):
            content = read_file_contents(file_path)
        elif describe_images and is_image_file(file_path):
            content = analyze_image_with_retries(file_path, gemini_model)
            images_described += 1
        elif describe_csv_files and is_csv_file(file_path):
            content = process_csv_file_with_retries(file_path, gemini_model)
            csvs_processed += 1
        elif process_notebooks and is_ipython_notebook(file_path):
            content, notebook_images = process_ipython_notebook_file(file_path, gemini_model, temp_dir=download_dir)
            images_described += notebook_images
            notebooks_processed += 1
        else:
            continue  # Skip other file types
        write_to_output(output_file, file_content_block(os.path.relpath(file_path, extracted_dir), content))

    # Cleanup downloaded repository folder
    shutil.rmtree(download_dir)

    print(f"Processed GitHub repository: {repo_url}")
    print(f"Output file created: {output_file}")
    print(f"Total files processed: {len(all_files) + (1 if readme_files else 0)}")
    if describe_images:
        print(f"Total images described: {images_described}")
    if describe_csv_files:
        print(f"Total CSV files processed: {csvs_processed}")
    if process_notebooks:
        print(f"Total notebooks processed: {notebooks_processed}")

    return images_described, csvs_processed, notebooks_processed


def process_local_folder(folder_path, output_dir, temp_dir='temp_repo_folder', describe_images=True, describe_csv_files=True, process_notebooks=True):
    """Process a local folder and create a dataset."""
    os.makedirs(temp_dir, exist_ok=True)
    folder_name = os.path.basename(folder_path)
    output_file = os.path.join(output_dir, f"{folder_name}.txt")

    # Write repo title and link
    date_string = datetime.now().strftime("%Y-%m-%d")
    starting_string = f"Folder title: {folder_name}\nFolder path: {folder_path}\nDate processed: {date_string}"
    write_to_output(output_file, delimiter_block(starting_string), mode='w')

    # Write file structure
    file_structure = generate_file_structure(folder_path)
    file_structure_content = '\n'.join(file_structure)
    write_to_output(output_file, file_content_block("folder file structure", file_structure_content))

    # Set up Gemini model if needed
    gemini_model = setup_gemini_model() if (describe_images or describe_csv_files or process_notebooks) else None

    # Process all files
    all_files = []
    images_described = 0
    csvs_processed = 0
    notebooks_processed = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not should_ignore(file_path) and (is_text_file(file_path) or 
                                                 (describe_images and is_image_file(file_path)) or 
                                                 (describe_csv_files and is_csv_file(file_path)) or
                                                 (process_notebooks and is_ipython_notebook(file_path))):
                all_files.append(file_path)

    # Find and process README file
    readme_files = [f for f in all_files if os.path.basename(f).lower().startswith('readme')]
    if readme_files:
        readme_file = readme_files[0]
        readme_content = read_file_contents(readme_file)
        write_to_output(output_file, file_content_block(os.path.relpath(readme_file, folder_path), readme_content))
        all_files.remove(readme_file)  # Remove README from all_files to avoid duplication

    # Process remaining files
    random.shuffle(all_files)
    for file_path in all_files:
        if is_text_file(file_path):
            content = read_file_contents(file_path)
        elif describe_images and is_image_file(file_path):
            content = analyze_image_with_retries(file_path, gemini_model)
            images_described += 1
        elif describe_csv_files and is_csv_file(file_path):
            content = process_csv_file_with_retries(file_path, gemini_model)
            csvs_processed += 1
        elif process_notebooks and is_ipython_notebook(file_path):
            content, notebook_images = process_ipython_notebook_file(file_path, gemini_model, temp_dir=temp_dir)
            images_described += notebook_images
            notebooks_processed += 1
        else:
            continue  # Skip other file types
        write_to_output(output_file, file_content_block(os.path.relpath(file_path, folder_path), content))

    # Cleanup downloaded repository folder
    shutil.rmtree(temp_dir)

    print(f"Processed local folder: {folder_path}")
    print(f"Output file created: {output_file}")
    print(f"Total files processed: {len(all_files) + (1 if readme_files else 0)}")
    if describe_images:
        print(f"Total images described: {images_described}")
    if describe_csv_files:
        print(f"Total CSV files processed: {csvs_processed}")
    if process_notebooks:
        print(f"Total notebooks processed: {notebooks_processed}")

    return images_described, csvs_processed, notebooks_processed


def convert_repos_to_text(sources, output_dir, is_github=True, describe_images=True, describe_csv_files=True, process_notebooks=True):
    """Create a dataset from multiple sources (GitHub repos or local folders)."""
    os.makedirs(output_dir, exist_ok=True)
    total_images_described = 0
    total_csvs_processed = 0
    total_notebooks_processed = 0

    for source in sources:
        try:
            if is_github:
                images_described, csvs_processed, notebooks_processed = process_github_repo(
                    source, output_dir, 
                    describe_images=describe_images, describe_csv_files=describe_csv_files, process_notebooks=process_notebooks)
            else:
                images_described, csvs_processed, notebooks_processed = process_local_folder(
                    source, output_dir, 
                    describe_images=describe_images, describe_csv_files=describe_csv_files, process_notebooks=process_notebooks)
            total_images_described += images_described
            total_csvs_processed += csvs_processed
            total_notebooks_processed += notebooks_processed
        except Exception as e:
            print(f"Error processing {source}: {str(e)}")

        print("=" * 50)

    if describe_images:
        print(f"Total images described across all sources: {total_images_described}")
    if describe_csv_files:
        print(f"Total CSV files processed across all sources: {total_csvs_processed}")
    if process_notebooks:
        print(f"Total notebooks processed across all sources: {total_notebooks_processed}")



#%% 