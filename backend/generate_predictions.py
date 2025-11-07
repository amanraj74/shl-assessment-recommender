import pandas as pd
from recommender import AssessmentRecommender
import os

print("🚀 Generating test set predictions...")

# Initialize recommender
recommender = AssessmentRecommender()

# Load test set
test_df = pd.read_excel('../data/Gen_AI-Dataset.xlsx', sheet_name='Test-Set')

print(f"📊 Found {len(test_df)} test queries")

# Generate predictions
predictions = []

for idx, row in test_df.iterrows():
    query = row['Query']
    
    # Get recommendations
    recommendations = recommender.get_recommendations(query, k=10)
    
    # Add each recommendation as a row
    for rec in recommendations:
        predictions.append({
            'Query': query,
            'Assessment_url': rec['url']
        })
    
    print(f"✅ Processed {idx + 1}/{len(test_df)} queries")

# Save predictions
output_df = pd.DataFrame(predictions)
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

output_file = f'{output_dir}/test_predictions.csv'
output_df.to_csv(output_file, index=False)

print(f"\n✨ Predictions saved to: {output_file}")
print(f"   Total predictions: {len(output_df)}")
print(f"\n📋 First few predictions:")
print(output_df.head(10))
