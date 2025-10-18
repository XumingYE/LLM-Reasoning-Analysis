
import json
import random

def find_anomalies(input_file, output_file, length_lower, length_upper, count_threshold, num_samples=5):
    anomalies = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                think_content = data.get('content', '').split('</think>')[0]
                think_content_length = len(think_content)
                
                counts = data.get('counts')
                
                # Check if the data point meets the new criteria
                if (length_lower <= think_content_length <= length_upper) and (counts <= count_threshold):
                    data['think_content_length'] = think_content_length
                    anomalies.append(data)
                    
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"Found {len(anomalies)} entries with {length_lower} <= think_content_length <= {length_upper} and counts <= {count_threshold}.")

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
    LENGTH_LOWER = 2000
    LENGTH_UPPER = 4000
    COUNT_THRESHOLD = 5
    find_anomalies(
        'data/merged_with_labels_and_counts.jsonl',
        'medium_think_low_counts_samples.jsonl',
        LENGTH_LOWER,
        LENGTH_UPPER,
        COUNT_THRESHOLD,
        num_samples=5
    )
