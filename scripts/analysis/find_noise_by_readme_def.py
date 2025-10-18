
import json
import sys
import os

def find_noise(input_file, output_dir):
    type1_noise = []
    type2_noise = []

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            content = data.get('content', '')
            counts = data.get('counts', 0)
            
            think_content = content.split('</think>')[0]
            think_content_length = len(think_content)
            
            # Type 1: Repetitive instruction noise
            if think_content_length >= 4000 and counts <= 5:
                data['think_content_length'] = think_content_length
                type1_noise.append(data)
            
            # Type 2: Content dislocation noise
            elif 2000 <= think_content_length < 4000 and counts <= 5:
                data['think_content_length'] = think_content_length
                type2_noise.append(data)

    # Save Type 1 noise samples
    type1_output_path = os.path.join(output_dir, 'qwen2_14b_type1_noise.jsonl')
    with open(type1_output_path, 'w') as f:
        for item in type1_noise:
            f.write(json.dumps(item) + '\n')
            
    # Save Type 2 noise samples
    type2_output_path = os.path.join(output_dir, 'qwen2_14b_type2_noise.jsonl')
    with open(type2_output_path, 'w') as f:
        for item in type2_noise:
            f.write(json.dumps(item) + '\n')
            
    print(f"Found {len(type1_noise)} samples of Type 1 noise (Repetitive Instruction).")
    print(f"Saved to {type1_output_path}")
    print(f"Found {len(type2_noise)} samples of Type 2 noise (Content Dislocation).")
    print(f"Saved to {type2_output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python find_noise_by_readme_def.py <input_file> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    find_noise(input_path, output_dir_path)
