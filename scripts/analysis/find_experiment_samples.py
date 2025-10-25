import json
import os
from collections import defaultdict
from transformers import AutoTokenizer

def find_experiment_samples(data_file, tokenizer_path, tolerance=5):
    """
    Finds data samples with similar content token length but different counts.

    Args:
        data_file (str): Path to the JSONL data file.
        tokenizer_path (str): Path to the tokenizer.
        tolerance (int): Tolerance for token count difference.
    """
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer path not found: {tokenizer_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    token_map = defaultdict(list)

    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                content = data.get("content", "")
                counts = data.get("counts", 0)
                
                if not content:
                    continue

                tokens = tokenizer.encode(content)
                token_count = len(tokens)
                
                token_map[token_count].append({"counts": counts, "data": data})

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {i+1}")
                continue

    sorted_token_counts = sorted(token_map.keys())

    # Use a set to keep track of printed sample groups to avoid duplicates
    printed_groups = set()

    for i in range(len(sorted_token_counts)):
        base_token_count = sorted_token_counts[i]
        
        # Create a tuple of the sorted data to make it hashable for the set
        base_samples_tuple = tuple(json.dumps(s['data']) for s in sorted(token_map[base_token_count], key=lambda x: x['data']['sample_id']))
        if base_samples_tuple in printed_groups:
            continue

        # Combine current bucket with buckets within the tolerance range
        combined_samples = list(token_map[base_token_count])
        for j in range(i + 1, len(sorted_token_counts)):
            next_token_count = sorted_token_counts[j]
            if next_token_count - base_token_count <= tolerance:
                combined_samples.extend(token_map[next_token_count])
            else:
                break

        if len(combined_samples) > 1:
            # Sort by counts to easily find differences
            combined_samples.sort(key=lambda x: x['counts'])
            
            min_counts_sample = combined_samples[0]
            max_counts_sample = combined_samples[-1]

            if min_counts_sample['counts'] != max_counts_sample['counts']:
                print(f"Found samples with similar token counts and different reasoning counts:")
                
                # Print the sample with the minimum counts
                min_tokens = tokenizer.encode(min_counts_sample['data'].get("content", ""))
                print(f"  - Token Count: {len(min_tokens)}, Counts: {min_counts_sample['counts']}, Sample ID: {min_counts_sample['data'].get('sample_id')}, Worker ID: {min_counts_sample['data'].get('worker_id')}")

                # Print the sample with the maximum counts
                max_tokens = tokenizer.encode(max_counts_sample['data'].get("content", ""))
                print(f"  - Token Count: {len(max_tokens)}, Counts: {max_counts_sample['counts']}, Sample ID: {max_counts_sample['data'].get('sample_id')}, Worker ID: {max_counts_sample['data'].get('worker_id')}")

                print(f"Found {len(combined_samples)} samples in this range.")
                print("-" * 20)

                # Add all samples in this combined group to printed_groups to avoid re-printing
                for sample in combined_samples:
                    sample_tuple = tuple(json.dumps(s['data']) for s in sorted([sample], key=lambda x: x['data']['sample_id']))
                    printed_groups.add(sample_tuple)


if __name__ == "__main__":
    DATA_FILE = "data/merged_with_labels_and_counts.jsonl"
    TOKENIZER_PATH = "/home/yexuming/model/hg/DeepSeek-R1-Distill-Qwen-14B"
    find_experiment_samples(DATA_FILE, TOKENIZER_PATH)