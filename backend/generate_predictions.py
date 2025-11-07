"""
V2.0 Test Set Predictions Generator - Complete Professional Version
Generates predictions on test set for submission
"""

import pandas as pd
from recommender import AssessmentRecommender
import os
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionGenerator:
    """Generate predictions on test set"""
    
    def __init__(self, test_file: str, recommender: AssessmentRecommender):
        """Initialize prediction generator"""
        self.recommender = recommender
        
        logger.info("📊 Loading test set...")
        self.test_df = pd.read_excel(test_file, sheet_name='Test-Set')
        logger.info(f"✅ Test set loaded: {len(self.test_df)} queries")
    
    def generate_predictions(self, k: int = 10) -> pd.DataFrame:
        """Generate predictions for all test queries"""
        logger.info(f"\n🚀 Generating predictions for {len(self.test_df)} test queries...")
        
        predictions = []
        
        for idx, row in self.test_df.iterrows():
            query = row['Query']
            
            # Get recommendations
            recommendations = self.recommender.get_recommendations(query, k=k)
            
            # Add each recommendation as a row
            for rec in recommendations:
                predictions.append({
                    'Query': query,
                    'Assessment_url': rec['url']
                })
            
            # Progress logging
            if (idx + 1) % max(1, len(self.test_df) // 5) == 0 or (idx + 1) == len(self.test_df):
                logger.info(f"   ✓ Processed {idx + 1}/{len(self.test_df)} queries")
        
        # Create DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"\n✅ Predictions generated")
        logger.info(f"   ├─ Total predictions: {len(predictions_df)}")
        logger.info(f"   ├─ Unique queries: {predictions_df['Query'].nunique()}")
        logger.info(f"   └─ Avg predictions per query: {len(predictions_df) / len(self.test_df):.1f}")
        
        return predictions_df
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_path: str) -> None:
        """Save predictions to CSV"""
        try:
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"✅ Predictions saved to: {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save predictions: {e}")
            raise

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("V2.0 TEST SET PREDICTION GENERATION")
    logger.info("="*70)
    
    try:
        # Initialize recommender
        logger.info("\n🏆 Initializing V2.0 Recommender...")
        recommender = AssessmentRecommender()
        
        # Initialize generator
        logger.info("\n📋 Initializing Prediction Generator...")
        generator = PredictionGenerator(
            '../data/Gen_AI-Dataset.xlsx',
            recommender
        )
        
        # Generate predictions
        predictions_df = generator.generate_predictions(k=10)
        
        # Save predictions
        output_dir = '../outputs'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/test_predictions.csv'
        generator.save_predictions(predictions_df, output_file)
        
        logger.info("\n" + "="*70)
        logger.info("✨ PREDICTION GENERATION COMPLETE")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        import traceback
        traceback.print_exc()
