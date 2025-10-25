
import pandas as pd
import re
import json
import argparse
from tqdm import tqdm

def get_think_content_length(text):
    """Extracts the length of the content within the <think> tags."""
    if not isinstance(text, str):
        return 0
    
    think_content_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not think_content_match:
        return 0
        
    return len(think_content_match.group(1))

def clean_data(input_path, output_path):
    """
    Reads a jsonl file, filters out noisy data based on specific criteria,
    and writes the clean data to an output jsonl file.
    
    Noise criteria: think_content_length >= 4000 AND counts <= 5
    """
    try:
        lines = []
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        clean_records = []
        removed_count = 0

        for line in tqdm(lines, desc="Cleaning data"):
            data = json.loads(line)
            
            think_length = get_think_content_length(data.get('content'))
            counts = data.get('counts', 0)
            
            # Noise condition
            is_noise = (think_length >= 4000 and counts <= 5)
            
            if not is_noise:
                clean_records.append(line) # Append the original line
            else:
                removed_count += 1

        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(clean_records)
                
        print(f"Processing complete.")
        print(f"Original records: {len(lines)}")
        print(f"Removed records: {removed_count}")
        print(f"Clean records saved: {len(clean_records)}")
        print(f"Clean data saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean noisy data from a jsonl file based on think content length and counts.")
    parser.add_argument('input_file', type=str, help='The path to the input jsonl file.')
    parser.add_argument('output_file', type=str, help='The path to the output jsonl file for the cleaned data.')
    
    args = parser.parse_args()
    
    clean_data(args.input_file, args.output_file)
