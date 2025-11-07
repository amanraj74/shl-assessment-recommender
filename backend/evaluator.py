import pandas as pd
import numpy as np
from recommender import AssessmentRecommender
from typing import List

class RecommendationEvaluator:
    def __init__(self, train_file: str, recommender: AssessmentRecommender):
        """Initialize evaluator with training data"""
        self.train_df = pd.read_excel(train_file, sheet_name='Train-Set')
        self.recommender = recommender
        
        # Group ground truth by query
        self.ground_truth = {}
        for query in self.train_df['Query'].unique():
            relevant_urls = set(
                self.train_df[self.train_df['Query'] == query]['Assessment_url'].tolist()
            )
            self.ground_truth[query] = relevant_urls
    
    def recall_at_k(self, predicted: List[str], relevant: set, k: int) -> float:
        """Calculate Recall@K"""
        predicted_k = set(predicted[:k])
        if len(relevant) == 0:
            return 0.0
        return len(predicted_k.intersection(relevant)) / len(relevant)
    
    def mean_average_precision_at_k(self, predicted: List[str], relevant: set, k: int) -> float:
        """Calculate Mean Average Precision@K"""
        score = 0.0
        hits = 0.0
        
        for i, pred_url in enumerate(predicted[:k]):
            if pred_url in relevant:
                hits += 1
                score += hits / (i + 1)
        
        if len(relevant) == 0:
            return 0.0
        
        return score / len(relevant)
    
    def evaluate(self, k: int = 10) -> dict:
        """Evaluate on all queries"""
        recalls = []
        maps = []
        results_detail = []
        
        for query, relevant_urls in self.ground_truth.items():
            # Get predictions
            recommendations = self.recommender.get_recommendations(query, k=k)
            predicted_urls = [rec['url'] for rec in recommendations]
            
            # Calculate metrics
            recall = self.recall_at_k(predicted_urls, relevant_urls, k)
            map_score = self.mean_average_precision_at_k(predicted_urls, relevant_urls, k)
            
            recalls.append(recall)
            maps.append(map_score)
            
            results_detail.append({
                'query': query[:50],
                'relevant': len(relevant_urls),
                'predicted': len(predicted_urls),
                f'recall@{k}': round(recall, 4),
                f'map@{k}': round(map_score, 4)
            })
        
        mean_recall = np.mean(recalls) if recalls else 0.0
        mean_map = np.mean(maps) if maps else 0.0
        
        return {
            f'mean_recall@{k}': mean_recall,
            f'mean_average_precision@{k}': mean_map,
            'total_queries': len(self.ground_truth),
            'details': pd.DataFrame(results_detail)
        }

if __name__ == "__main__":
    recommender = AssessmentRecommender()
    evaluator = RecommendationEvaluator('../data/Gen_AI-Dataset.xlsx', recommender)
    
    print("\n" + "="*70)
    print("EVALUATING RECOMMENDATION SYSTEM")
    print("="*70)
    
    # Evaluate at different k values
    for k in [3, 5, 10]:
        results = evaluator.evaluate(k=k)
        
        print(f"\n📊 Results at k={k}:")
        print(f"   Mean Recall@{k}: {results[f'mean_recall@{k}']:.4f}")
        print(f"   Mean Average Precision@{k}: {results[f'mean_average_precision@{k}']:.4f}")
        print(f"   Total Queries: {results['total_queries']}")
