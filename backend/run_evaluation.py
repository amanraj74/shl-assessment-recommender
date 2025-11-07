"""
V2.0 Evaluation Script - Complete Professional Version
Calculates Recall@K and MAP@K metrics on training ground truth
"""

import pandas as pd
import numpy as np
from recommender import AssessmentRecommender
import json
import os
import logging
from typing import Set, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    """Professional evaluation using ground truth"""
    
    def __init__(self, train_file: str, recommender: AssessmentRecommender):
        """Initialize evaluator"""
        logger.info("\n" + "="*70)
        logger.info("V2.0 EVALUATION ENGINE - INITIALIZING")
        logger.info("="*70)
        
        self.recommender = recommender
        
        logger.info("\n📊 Loading ground truth training data...")
        self.train_df = pd.read_excel(train_file, sheet_name='Train-Set')
        
        # Build ground truth mapping: query -> set of relevant assessment URLs
        self.ground_truth = {}
        for query in self.train_df['Query'].unique():
            relevant_urls = set(
                self.train_df[self.train_df['Query'] == query]['Assessment_url'].tolist()
            )
            self.ground_truth[query] = relevant_urls
        
        logger.info(f"✅ Ground truth loaded")
        logger.info(f"   ├─ Unique queries: {len(self.ground_truth)}")
        logger.info(f"   ├─ Total examples: {len(self.train_df)}")
        logger.info(f"   └─ Average relevant per query: {np.mean([len(v) for v in self.ground_truth.values()]):.1f}")
    
    def recall_at_k(self, predicted: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Recall@K
        Proportion of relevant assessments that appear in top-k predictions
        
        Recall@K = |predicted_top_k ∩ relevant| / |relevant|
        """
        predicted_k = set(predicted[:k])
        
        if len(relevant) == 0:
            return 0.0
        
        hits = len(predicted_k.intersection(relevant))
        recall = hits / len(relevant)
        
        return recall
    
    def mean_average_precision_at_k(self, predicted: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Mean Average Precision@K
        Rewards relevant items appearing early in ranking
        
        MAP@K = (1/|relevant|) * Σ(P(i) * rel(i)) where i ∈ [1..k]
        """
        score = 0.0
        hits = 0.0
        
        for i, pred_url in enumerate(predicted[:k], 1):
            if pred_url in relevant:
                hits += 1
                precision_at_i = hits / i
                score += precision_at_i
        
        if len(relevant) == 0:
            return 0.0
        
        mean_avg_precision = score / len(relevant)
        return mean_avg_precision
    
    def evaluate_at_k(self, k: int) -> Dict:
        """
        Evaluate recommendation system at given k value
        
        Returns:
            Dictionary with mean_recall@k and mean_map@k
        """
        logger.info(f"\n{'─'*70}")
        logger.info(f"EVALUATING AT k={k}")
        logger.info(f"{'─'*70}")
        
        recalls = []
        maps = []
        results_per_query = []
        
        total_queries = len(self.ground_truth)
        
        for query_idx, (query, relevant_urls) in enumerate(self.ground_truth.items(), 1):
            # Get recommendations
            recommendations = self.recommender.get_recommendations(query, k=k)
            predicted_urls = [rec['url'] for rec in recommendations]
            
            # Calculate metrics
            recall = self.recall_at_k(predicted_urls, relevant_urls, k)
            map_score = self.mean_average_precision_at_k(predicted_urls, relevant_urls, k)
            
            recalls.append(recall)
            maps.append(map_score)
            
            results_per_query.append({
                'query': query[:60],
                'relevant_count': len(relevant_urls),
                'recall': recall,
                'map': map_score
            })
            
            # Print progress
            if query_idx % max(1, total_queries // 5) == 0 or query_idx == total_queries:
                logger.info(f"   ✓ Processed {query_idx}/{total_queries} queries")
        
        # Calculate means
        mean_recall = np.mean(recalls)
        mean_map = np.mean(maps)
        
        # Print summary
        logger.info(f"\n📊 Results at k={k}:")
        logger.info(f"   ├─ Mean Recall@{k}: {mean_recall:.4f} ({mean_recall*100:.2f}%)")
        logger.info(f"   ├─ Mean MAP@{k}: {mean_map:.4f} ({mean_map*100:.2f}%)")
        logger.info(f"   ├─ Total queries: {len(self.ground_truth)}")
        logger.info(f"   └─ Target (Recall>0.70, MAP>0.60): {'✅ EXCEEDED' if mean_recall > 0.70 and mean_map > 0.60 else '❌ BELOW'}")
        
        return {
            f'mean_recall@{k}': mean_recall,
            f'mean_map@{k}': mean_map,
            'total_queries': len(self.ground_truth),
            'results_per_query': results_per_query
        }

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("V2.0 SYSTEM EVALUATION - CHAMPIONSHIP PERFORMANCE TEST")
    logger.info("="*70)
    
    try:
        # Initialize recommender
        logger.info("\n🏆 Initializing V2.0 Recommender...")
        recommender = AssessmentRecommender()
        
        # Initialize evaluator
        logger.info("\n📊 Initializing Evaluator...")
        evaluator = RecommendationEvaluator(
            '../data/Gen_AI-Dataset.xlsx',
            recommender
        )
        
        # Evaluate at different k values
        logger.info("\n" + "="*70)
        logger.info("RUNNING EVALUATION TESTS")
        logger.info("="*70)
        
        results = {}
        for k in [3, 5, 10]:
            result = evaluator.evaluate_at_k(k=k)
            results[f'k={k}'] = result
        
        # Print final summary
        logger.info("\n" + "="*70)
        logger.info("FINAL EVALUATION SUMMARY")
        logger.info("="*70)
        
        for k_key, metrics in results.items():
            k = k_key.split('=')[1]
            recall_key = f'mean_recall@{k}'
            map_key = f'mean_map@{k}'
            
            logger.info(f"\n{k_key.upper()}:")
            logger.info(f"   Recall@{k}: {metrics[recall_key]:.4f} ({metrics[recall_key]*100:.2f}%)")
            logger.info(f"   MAP@{k}: {metrics[map_key]:.4f} ({metrics[map_key]*100:.2f}%)")
        
        # Save results
        output_dir = '../outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create simplified results for submission
        simplified_results = {}
        for k_key, metrics in results.items():
            simplified_results[k_key] = {
                f'mean_recall@{k_key.split("=")[1]}': metrics[f'mean_recall@{k_key.split("=")[1]}'],
                f'mean_map@{k_key.split("=")[1]}': metrics[f'mean_map@{k_key.split("=")[1]}'],
                'total_queries': metrics['total_queries']
            }
        
        output_file = f'{output_dir}/evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(simplified_results, f, indent=4)
        
        logger.info(f"\n💾 Results saved to: {output_file}")
        logger.info("\n" + "="*70)
        logger.info("✨ EVALUATION COMPLETE")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
