
import json
import re
import sys

def process_file(input_file, output_file):
    keywords = [
        'Wait', 'Alternatively', 'Another angle', 'Another approach', 'But wait',
        'Hold on', 'Hmm', 'Maybe', 'Looking back', 'Okay', 'Let me', 'First', 'Then',
        'Alright', 'Compute', 'Correct', 'Good', 'Got it', 'I don\'t see any errors', 'I think',
        'Let me double-check', 'Let\'s see', 'Now', 'Remember', 'Seems solid', 'Similarly',
        'So', 'Starting', 'That\'s correct', 'That seems right', 'Therefore', 'Thus'
    ]
    
    # Create a single regex pattern to find any of the keywords
    keyword_pattern = re.compile(r'(' + '|'.join(re.escape(k) for k in keywords) + r')')

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                content = data.get('content', '')
                
                # Focus on content before the </think> tag
                think_content = content.split('</think>')[0]
                
                # Count occurrences of all keywords
                num_keywords = len(keyword_pattern.findall(think_content))
                
                # counts = n + 1
                data['counts'] = num_keywords + 1
                
                outfile.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON
                pass

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python add_counts_field.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process_file(input_path, output_path)
    print(f"Processing complete. Output saved to {output_path}")
