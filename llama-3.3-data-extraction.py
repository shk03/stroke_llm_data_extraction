import os
import pandas as pd
from llama_cpp import Llama
import time
import json

def read_file(file_path):
    """
    Safely reads the content of a file with error handling.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        str or None: File content as a string if successful, otherwise None.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def construct_prompt(json_schema, annotation_guidelines, report_text):
    """
    Constructs the prompt for the LLM model using a template.
    
    Args:
        json_schema (dict): JSON schema structure for output.
        annotation_guidelines (str): Guidelines for annotation.
        report_text (str): Radiological report text.
    
    Returns:
        str: The constructed prompt.
    """
    prompt_template = f"""
    Extrahiere die im radiologischen Befundbericht gegebenen informationen im Format einer JSON Datei.
    Jeder der zehn Parameter soll mit "Ja" oder "Nein" gewertet werden.
    Befunde, die nicht erwähnt werden, gelten als nicht vorhanden und sollten mit "Nein" gewertet werden.

    Beachte dabei die folgenden Hinweise:    
    {annotation_guidelines} 

    Die JSON-Datei soll folgende Struktur haben:
    {json_schema}

    Hier ist der Befundbericht, aus dem die Informationen extrahiert werden sollen:
    {report_text}
    """
    return prompt_template


def main():
    """
    Main function to execute the script: load model, process files, 
    generate JSON output, and save results.
    """
    # Start timer
    start_time = time.time()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Script started at: {start_timestamp}")

    # Paths (Adjust these paths as needed)
    model_folder = "/path/to/model/folder"
    llm_model = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    output_folder = "/path/to/output/folder"
    folder_path = "/path/to/report/folder"

    # Load the LLM model
    llm = Llama(
        model_path=os.path.join(model_folder, llm_model),
        chat_format="llama-3",
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False
    )

    # Load JSON schema template
    try:
        with open('path/to/template', 'r') as file:
            json_schema = json.load(file)
    except Exception as e:
        print(f"Error loading JSON schema: {e}")
        return

    # Load annotation guidelines
    try:
        with open('path/to/annotation_guideline', 'r', encoding='utf-8') as file:
            annotation_guidelines = file.read()
    except Exception as e:
        print(f"Error loading annotation guidelines: {e}")
        return

    # Process text files in the folder
    txt_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.txt')])

    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        file_content = read_file(file_path)

        if not file_content:
            print(f"Skipping {file} due to read error")
            continue

        try:
            # Construct prompt and get LLM response
            prompt = construct_prompt(json_schema, annotation_guidelines, file_content)
            print(f"Processing {file}")

            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant tasked with extracting data parameters from text. "
                            "Output must strictly follow the JSON schema provided below. Any deviation is not acceptable. "
                            "Output only the JSON object and nothing else."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.0
            )

            # Parse the LLM response and save results
            message_content = response["choices"][0]["message"]["content"].strip()
            try:
                json_object = json.loads(message_content.strip('```').strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {file}: {e}")
                print(f"Problematic content: {repr(message_content)}")
                continue

            output_path = os.path.join(output_folder, f"{file.replace('.txt', '')}.json")
            with open(output_path, "w") as json_file:
                json.dump(json_object, json_file, indent=4)
            print(f"JSON saved to: {output_path}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # End timer
    end_time = time.time()
    end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    duration = end_time - start_time

    print(f"Script ended at: {end_timestamp}")
    print(f"Total duration: {duration:.2f} seconds")

    time_report = (
        f"Script started at: {start_timestamp}\n"
        f"Script ended at: {end_timestamp}\n"
        f"Total duration: {duration:.2f} seconds"
    )

    with open("/path/to/time_report.txt", 'w', encoding="utf-8") as file:
        file.write(time_report)


if __name__ == "__main__":
    main()
