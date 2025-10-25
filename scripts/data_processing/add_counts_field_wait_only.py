
import pandas as pd
import re
import json
import argparse
from tqdm import tqdm

def calculate_wait_counts(text):
    """
    Calculates the 'counts' based only on the case-sensitive word 'Wait'.
    The formula is count('Wait') + 1.
    """
    if not isinstance(text, str):
        return 1
    
    # Extract content within <think> tags
    think_content_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not think_content_match:
        return 1
        
    think_content = think_content_match.group(1)
    
    # Count the occurrences of the case-sensitive word "Wait"
    # We use regex with word boundary \b to ensure we match the whole word
    wait_count = len(re.findall(r'\bWait\b', think_content))
    
    return wait_count + 1

def process_file(input_path, output_path):
    """
    Reads a jsonl file, calculates the 'counts' field based on 'Wait' word,
    and writes the new data to an output jsonl file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            lines = infile.readlines()
            for line in tqdm(lines, desc="Processing file"):
                data = json.loads(line)
                
                # Calculate new counts based on 'content' field
                data['counts'] = calculate_wait_counts(data.get('content'))
                
                # Write the updated record to the output file
                outfile.write(json.dumps(data) + '\n')
                
        print(f"Successfully processed {len(lines)} records.")
        print(f"New file saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate 'counts' based on 'Wait' occurrences in the think content.")
    parser.add_argument('input_file', type=str, help='The path to the input jsonl file.')
    parser.add_argument('output_file', type=str, help='The path to the output jsonl file.')
    
    args = parser.parse_args()
    
    process_file(args.input_file, args.output_file)
