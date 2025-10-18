
import json
import random

def find_anomalies(input_file, output_file, count_threshold, length_threshold, num_samples=5):
    anomalies = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                think_content = data.get('content', '').split('</think>')[0]
                think_content_length = len(think_content)
                
                counts = data.get('counts')
                
                # Check for high counts and low length
                if (counts >= count_threshold) and (think_content_length <= length_threshold):
                    data['think_content_length'] = think_content_length
                    anomalies.append(data)
                    
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"Found {len(anomalies)} entries with counts >= {count_threshold} and think_content_length <= {length_threshold}.")

    if anomalies:
        if len(anomalies) > num_samples:
            sampled_data = random.sample(anomalies, num_samples)
        else:
            sampled_data = anomalies
            
        with open(output_file, 'w') as f:
            for item in sampled_data:
                f.write(json.dumps(item, indent=2) + '\n')
        
        print(f"Saved {len(sampled_data)} random samples to {output_file}")

if __name__ == '__main__':
    COUNT_THRESHOLD = 10  # High counts
    LENGTH_THRESHOLD = 100 # Low length
    find_anomalies(
        'data/merged_with_labels_and_counts.jsonl',
        'high_counts_low_length_samples.jsonl',
        COUNT_THRESHOLD,
        LENGTH_THRESHOLD,
        num_samples=5
    )

