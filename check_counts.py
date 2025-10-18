
import json
import random

def get_low_count_samples(input_file, output_file, count_value, num_samples=10):
    low_count_samples = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('counts') == count_value:
                    low_count_samples.append(data)
            except (json.JSONDecodeError, KeyError):
                pass

    if len(low_count_samples) > num_samples:
        sampled_data = random.sample(low_count_samples, num_samples)
    else:
        sampled_data = low_count_samples

    if sampled_data:
        with open(output_file, 'w') as f:
            for item in sampled_data:
                f.write(json.dumps(item) + '\n')

    print(f"Found {len(low_count_samples)} entries with counts={count_value}.")
    if sampled_data:
        print(f"Saved {len(sampled_data)} random samples to {output_file}")

if __name__ == '__main__':
    print("Checking for counts=0...")
    get_low_count_samples('data/merged_with_labels_and_counts.jsonl', 'counts_zero_samples_check.jsonl', count_value=0, num_samples=5)
    print("\nChecking for counts=1...")
    get_low_count_samples('data/merged_with_labels_and_counts.jsonl', 'counts_one_samples.jsonl', count_value=1, num_samples=10)
