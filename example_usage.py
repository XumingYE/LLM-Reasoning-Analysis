#!/usr/bin/env python3
"""
Example: Using the cluster-based model in a production application
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


class ComplexityPredictor:
    """Predict reasoning complexity and task type for user queries"""

    def __init__(self, model_path='predictor_deberta_clusters_clean'):
        """Load model and configuration"""
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # Load cluster configuration
        with open('data/cluster_config_clean.json', 'r') as f:
            self.config = json.load(f)

        # Label descriptions
        self.label_info = {
            0: {"type": "General", "complexity": "Simple", "avg_counts": 9.6},
            1: {"type": "Explanation", "complexity": "Simple", "avg_counts": 9.8},
            2: {"type": "Explanation", "complexity": "Moderate", "avg_counts": 10.7},
            3: {"type": "General", "complexity": "Moderate", "avg_counts": 11.0},
            4: {"type": "General", "complexity": "Moderate", "avg_counts": 11.4},
            5: {"type": "Creative", "complexity": "Moderate", "avg_counts": 12.0},
            6: {"type": "Coding/Calculation", "complexity": "Complex", "avg_counts": 15.0},
            7: {"type": "Long-form", "complexity": "Complex", "avg_counts": 16.9},
        }

        print("Model loaded successfully!")

    def extract_features(self, text):
        """Extract features for enhanced prediction"""
        word_count = len(text.split())

        # Detect task types
        has_explain = any(w in text.lower() for w in ['explain', 'describe', 'analyze', 'compare'])
        has_creative = any(w in text.lower() for w in ['write', 'create', 'generate', 'story'])
        has_calculate = any(w in text.lower() for w in ['calculate', 'compute', 'solve'])
        has_list = any(w in text.lower() for w in ['list', 'enumerate'])
        has_why = 'why' in text.lower()
        has_how = 'how' in text.lower()

        # Build feature tokens
        features = []
        features.append('[SHORT]' if word_count < 10 else '[MEDIUM]' if word_count < 25 else '[LONG]')
        if has_explain: features.append('[EXPLAIN]')
        if has_creative: features.append('[CREATIVE]')
        if has_calculate: features.append('[CALCULATE]')
        if has_list: features.append('[LIST]')
        if has_why: features.append('[WHY]')
        if has_how: features.append('[HOW]')

        return ' '.join(features) + ' ' + text if features else text

    def predict(self, query):
        """
        Predict complexity and task type for a query

        Returns:
            dict: {
                'label': int,
                'task_type': str,
                'complexity': str,
                'confidence': float,
                'expected_counts': float,
                'all_probabilities': dict
            }
        """
        # Enhance query with features
        enhanced_query = self.extract_features(query)

        # Tokenize
        inputs = self.tokenizer(enhanced_query, return_tensors="pt",
                                padding=True, truncation=True, max_length=512)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            predicted_label = int(torch.argmax(logits, dim=-1).item())

        # Build result
        label_data = self.label_info[predicted_label]

        return {
            'label': predicted_label,
            'task_type': label_data['type'],
            'complexity': label_data['complexity'],
            'confidence': float(probs[predicted_label]),
            'expected_counts': label_data['avg_counts'],
            'all_probabilities': {i: float(probs[i]) for i in range(len(probs))}
        }

    def route_query(self, query):
        """
        Route query to appropriate handler based on prediction

        Returns:
            str: Handler name
        """
        result = self.predict(query)

        # Routing logic
        if result['task_type'] == 'Explanation':
            return 'analytical_pipeline'
        elif result['task_type'] == 'Creative':
            return 'generation_pipeline'
        elif result['task_type'] == 'Coding/Calculation':
            return 'code_execution_pipeline'
        elif result['complexity'] == 'Complex':
            return 'high_resource_pipeline'
        else:
            return 'standard_pipeline'

    def allocate_resources(self, query):
        """
        Determine resource allocation based on complexity

        Returns:
            dict: Resource configuration
        """
        result = self.predict(query)

        if result['complexity'] == 'Complex':
            return {
                'cpu_cores': 4,
                'memory_gb': 8,
                'timeout_seconds': 300,
                'priority': 'high'
            }
        elif result['complexity'] == 'Moderate':
            return {
                'cpu_cores': 2,
                'memory_gb': 4,
                'timeout_seconds': 120,
                'priority': 'normal'
            }
        else:  # Simple
            return {
                'cpu_cores': 1,
                'memory_gb': 2,
                'timeout_seconds': 60,
                'priority': 'low'
            }


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ComplexityPredictor()

    # Example queries
    test_queries = [
        "What are the three primary colors?",
        "Write a short story about a dragon",
        "Explain how neural networks work in deep learning",
        "Calculate the derivative of x^2 + 3x + 5",
        "Compare and contrast democracy and autocracy",
        "List five benefits of regular exercise",
    ]

    print("\n" + "="*70)
    print("COMPLEXITY PREDICTION EXAMPLES")
    print("="*70 + "\n")

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 70)

        # Predict
        result = predictor.predict(query)
        print(f"  Type: {result['task_type']}")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Expected reasoning steps: ~{result['expected_counts']:.0f}")

        # Route
        handler = predictor.route_query(query)
        print(f"  → Routed to: {handler}")

        # Resources
        resources = predictor.allocate_resources(query)
        print(f"  → Resources: {resources['cpu_cores']} cores, {resources['memory_gb']}GB RAM, "
              f"{resources['timeout_seconds']}s timeout")

        print()

    print("="*70)
    print("\n✓ Predictions complete!")
