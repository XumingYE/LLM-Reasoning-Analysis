
import json
import random

def get_zero_count_samples(input_file, output_file, num_samples=10):
    zero_count_samples = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('counts') == 0:
                zero_count_samples.append(data)
    
    if len(zero_count_samples) > num_samples:
        sampled_data = random.sample(zero_count_samples, num_samples)
    else:
        sampled_data = zero_count_samples
        
    with open(output_file, 'w') as f:
        for item in sampled_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Found {len(zero_count_samples)} entries with counts=0.")
    print(f"Saved {len(sampled_data)} random samples to {output_file}")

if __name__ == '__main__':
    get_zero_count_samples('data/merged_with_labels_and_counts.jsonl', 'counts_zero_samples.jsonl')

