import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict
import json

class EmbeddingsGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with sentence transformer model
        Options: 
        - 'all-MiniLM-L6-v2' (fast, good quality)
        - 'all-mpnet-base-v2' (slower, better quality)
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def create_assessment_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Create embeddings for all assessments"""
        
        # Combine multiple fields for rich representation
        texts = []
        for _, row in df.iterrows():
            # Create comprehensive text representation
            text = f"""
            Assessment: {row['name']}
            Description: {row['description']}
            Skills: {', '.join(row['skills']) if isinstance(row['skills'], list) else row['skills']}
            Test Type: {row['test_type']}
            Duration: {row['duration']} minutes
            """
            texts.append(text.strip())
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for fast similarity search"""
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine similarity
        index.add(embeddings)
        
        return index
    
    def save_artifacts(self, embeddings: np.ndarray, index: faiss.Index, 
                      df: pd.DataFrame, output_dir: str):
        """Save all artifacts for later use"""
        
        # Save embeddings
        np.save(f'{output_dir}/embeddings.npy', embeddings)
        
        # Save FAISS index
        faiss.write_index(index, f'{output_dir}/faiss_index.bin')
        
        # Save assessment metadata
        df.to_csv(f'{output_dir}/assessments_metadata.csv', index=False)
        
        print(f"Saved artifacts to {output_dir}")

if __name__ == "__main__":
    # Load crawled data
    df = pd.read_csv('../data/raw/shl_assessments.csv')
    
    # Generate embeddings
    generator = EmbeddingsGenerator()
    embeddings = generator.create_assessment_embeddings(df)
    
    # Build FAISS index
    index = generator.build_faiss_index(embeddings)
    
    # Save everything
    generator.save_artifacts(embeddings, index, df, '../data/processed')
