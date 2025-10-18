import json
import sys
import pandas as pd
import os

def find_high_counts(input_file, output_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    
    # Sort by counts and get the top 5
    top_5_high_counts = df.nlargest(5, 'counts')
    
    # Save to output file
    with open(output_file, 'w') as f:
        for i, row in top_5_high_counts.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
            
    print(f"Saved top 5 highest counts samples to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python find_high_counts.py <input_file> <output_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    find_high_counts(input_path, output_path)