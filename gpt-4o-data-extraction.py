"""
Version: pip install openai==0.28.0
"""

import openai
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# ------------- Step 1: Load environment variables -------------
# The API key must be stored in a .env file in the project directory.
# Example .env file content:
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

load_dotenv("path/to/your/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------- Step 2: Function to extract structured data from reports -------------
def extract_data_from_report(report_text, json_schema, annotation_guidelines):
    """

    Args:
        report_text (str): The complete radiology report to be analyzed.
        json_schema (str): The JSON schema that defines the expected output structure.
        annotation_guidelines (str): Additional instructions and guidelines for data extraction.

    Returns:
        dict or None: Extracted JSON data as a dictionary if successful, otherwise None.
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

    # Send the request to the OpenAI API
    response = openai.ChatCompletion.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "You are a data extraction assistant."},
          {"role": "user", "content": prompt_template}
      ],
      response_format={"type": "json_object"}
    )

    extracted_data = response.choices[0].message.content

    try:
        return json.loads(extracted_data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw response: {extracted_data}")
        return None

# ------------- Step 3: Main function to process multiple reports -------------
def main():
    """
    Main function that processes multiple radiology reports from a specified directory,
    extracts structured data, and saves the results as JSON files in an output directory.
    """

    # Paths to the input and output directories
    REPORTS_DIR = "path/to/report/folder"
    JSON_SCHEMA_PATH = "path/to/template"
    ANNOTATION_GUIDELINES_PATH = "path/to/annotation_guideline"

    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR = f"output_{timestamp}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the schema and annotation guidelines
    with open(JSON_SCHEMA_PATH, 'r', encoding='utf-8') as f:
        json_schema = f.read()

    with open(ANNOTATION_GUIDELINES_PATH, 'r', encoding='utf-8') as f:
        annotation_guidelines = f.read()

    # Process each report in the reports directory
    start_time = time.time()

    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith('.txt'):
            report_path = os.path.join(REPORTS_DIR, filename)

            with open(report_path, 'r', encoding='utf-8') as file:
                report_text = file.read()

            structured_report = extract_data_from_report(report_text, json_schema, annotation_guidelines)

            # Save the structured data to a JSON file
            output_filename = f"{os.path.splitext(filename)[0]}.json"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_filepath, 'w', encoding='utf-8') as json_file:
                json.dump(structured_report, json_file, indent=4, ensure_ascii=False)

            print(f"Processed and saved: {output_filename}")

    end_time = time.time()
    duration = end_time - start_time

    # Log the execution time
    time_log_filename = os.path.join(OUTPUT_DIR, f"{timestamp}_execution_time_log.txt")
    with open(time_log_filename, "w", encoding='utf-8') as log_file:
        log_file.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Execution Duration: {duration:.2f} seconds\n")

    print(f"Execution time logged in '{time_log_filename}'.")
    print("Extraction and saving completed.")

# ------------- Step 4: Run the script -------------
if __name__ == "__main__":
    main()