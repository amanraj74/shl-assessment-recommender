import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

print("🚀 Starting data preparation...")

# Create directories if they don't exist
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../data/raw', exist_ok=True)

# Load the training dataset
print("📊 Loading training dataset...")
train_df = pd.read_excel('../data/Gen_AI-Dataset.xlsx', sheet_name='Train-Set')

# Get unique assessments
print(f"📈 Found {len(train_df)} training examples")
print(f"📈 Unique queries: {train_df['Query'].nunique()}")
print(f"📈 Unique assessments: {train_df['Assessment_url'].nunique()}")

# Create assessment metadata from training data
# Group by URL to get unique assessments
assessments = train_df.groupby('Assessment_url').first().reset_index()
assessments = assessments[['Assessment_url']].copy()
assessments.columns = ['url']

# Extract assessment names from URLs (basic extraction)
assessments['name'] = assessments['url'].apply(
    lambda x: x.split('/')[-2].replace('-', ' ').title() if '/' in x else 'Unknown'
)

# For now, we'll use the URL as description (we'll enhance this later)
assessments['description'] = assessments['url']
assessments['test_type'] = 'Unknown'
assessments['duration'] = 0
assessments['skills'] = ''

print(f"\n✅ Extracted {len(assessments)} unique assessments")

# Save metadata
assessments.to_csv('../data/processed/assessments_metadata.csv', index=False)
print("💾 Saved assessments metadata")

# Generate embeddings
print("\n🧠 Generating embeddings (this may take a minute)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create text representations for embeddings
texts = []
for _, row in assessments.iterrows():
    text = f"Assessment: {row['name']}\nURL: {row['url']}\nDescription: {row['description']}"
    texts.append(text)

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)
print(f"✅ Generated embeddings with shape: {embeddings.shape}")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Save embeddings
np.save('../data/processed/embeddings.npy', embeddings)
print("💾 Saved embeddings")

# Create FAISS index
print("\n🔍 Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, '../data/processed/faiss_index.bin')
print("💾 Saved FAISS index")

print("\n✨ Data preparation complete!")
print("\n📁 Created files:")
print("   - ../data/processed/assessments_metadata.csv")
print("   - ../data/processed/embeddings.npy")
print("   - ../data/processed/faiss_index.bin")
print("\n🎉 You can now run: python app.py")
