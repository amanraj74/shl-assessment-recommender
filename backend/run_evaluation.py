import pandas as pd
import numpy as np
from recommender import AssessmentRecommender
import json
import os

print("\n" + "="*70)
print("EVALUATING RECOMMENDATION SYSTEM")
print("="*70)

# Initialize recommender
recommender = AssessmentRecommender()

# Load training data
train_df = pd.read_excel('../data/Gen_AI-Dataset.xlsx', sheet_name='Train-Set')

print(f"\n📊 Training data loaded: {len(train_df)} examples")

# Group by query to get ground truth
ground_truth = {}
for query in train_df['Query'].unique():
    relevant_urls = set(train_df[train_df['Query'] == query]['Assessment_url'].tolist())
    ground_truth[query] = relevant_urls

print(f"📈 Unique queries: {len(ground_truth)}")
print(f"📈 Unique assessments: {len(train_df['Assessment_url'].unique())}")

# Evaluation metrics
def recall_at_k(predicted, relevant, k):
    """Calculate Recall@K"""
    predicted_k = set(predicted[:k])
    if len(relevant) == 0:
        return 0.0
    return len(predicted_k.intersection(relevant)) / len(relevant)

def mean_average_precision_at_k(predicted, relevant, k):
    """Calculate MAP@K"""
    score = 0.0
    hits = 0.0
    
    for i, pred_url in enumerate(predicted[:k]):
        if pred_url in relevant:
            hits += 1
            score += hits / (i + 1)
    
    if len(relevant) == 0:
        return 0.0
    
    return score / len(relevant)

# Evaluate at different k values
results = {}

for k in [3, 5, 10]:
    print(f"\n{'─'*70}")
    print(f"Evaluating at k={k}")
    print('─'*70)
    
    recalls = []
    maps = []
    details = []
    
    for query_idx, (query, relevant_urls) in enumerate(ground_truth.items(), 1):
        # Get predictions
        recommendations = recommender.get_recommendations(query, k=k)
        predicted_urls = [rec['url'] for rec in recommendations]
        
        # Calculate metrics
        recall = recall_at_k(predicted_urls, relevant_urls, k)
        map_score = mean_average_precision_at_k(predicted_urls, relevant_urls, k)
        
        recalls.append(recall)
        maps.append(map_score)
        
        details.append({
            'query': query[:50] + '...' if len(query) > 50 else query,
            'relevant_count': len(relevant_urls),
            'predicted_count': len(predicted_urls),
            f'recall@{k}': round(recall, 4),
            f'map@{k}': round(map_score, 4)
        })
        
        if query_idx % 2 == 0:
            print(f"   ✓ Processed {query_idx}/{len(ground_truth)} queries")
    
    mean_recall = np.mean(recalls)
    mean_map = np.mean(maps)
    
    results[f'k={k}'] = {
        f'mean_recall@{k}': round(mean_recall, 4),
        f'mean_map@{k}': round(mean_map, 4),
        'total_queries': len(ground_truth),
        'details': details
    }
    
    print(f"\n📊 Results at k={k}:")
    print(f"   Mean Recall@{k}: {mean_recall:.4f}")
    print(f"   Mean MAP@{k}: {mean_map:.4f}")
    print(f"   Queries evaluated: {len(ground_truth)}")

# Print summary
print("\n" + "="*70)
print("FINAL EVALUATION SUMMARY")
print("="*70)

for k_result, metrics in results.items():
    print(f"\n{k_result.upper()}:")
    recall_key = f"mean_recall@{k_result.split('=')[1]}"
    map_key = f"mean_map@{k_result.split('=')[1]}"
    print(f"  ✓ {recall_key}: {metrics[recall_key]:.4f}")
    print(f"  ✓ {map_key}: {metrics[map_key]:.4f}")

# Save results
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

results_file = f'{output_dir}/evaluation_results.json'
with open(results_file, 'w') as f:
    # Convert for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            k2: v2 for k2, v2 in v.items() if k2 != 'details'
        }
    json.dump(json_results, f, indent=4)

print(f"\n✅ Results saved to: {results_file}")
print("="*70 + "\n")
