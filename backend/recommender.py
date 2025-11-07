import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Set
import os
from sklearn.metrics.pairwise import cosine_similarity

class AssessmentRecommender:
    def __init__(self, data_dir: str = '../data/processed'):
        """Initialize advanced recommender with enhanced features"""
        
        try:
            # Load embeddings and index
            self.embeddings = np.load(f'{data_dir}/embeddings.npy')
            self.index = faiss.read_index(f'{data_dir}/faiss_index.bin')
            self.metadata = pd.read_csv(f'{data_dir}/assessments_metadata.csv')
            
            # Load embedding model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Build skill dictionary for smarter matching
            self.skill_index = self._build_skill_index()
            
            print(f"✅ Loaded {len(self.metadata)} assessments")
            print(f"✅ Embedding dimension: {self.embeddings.shape[1]}")
            print(f"✅ Skill index created with {len(self.skill_index)} unique skills")
            
        except Exception as e:
            print(f"❌ Error initializing recommender: {e}")
            raise
    
    def _build_skill_index(self) -> Dict[str, Set[int]]:
        """Build index of skills to assessment IDs for faster lookup"""
        skill_index = {}
        
        # Comprehensive skill dictionary
        skill_keywords = {
            'python': ['python', 'py'],
            'java': ['java'],
            'sql': ['sql', 'database', 'sql'],
            'javascript': ['javascript', 'js', 'node', 'react', 'angular', 'vue'],
            'c++': ['c++', 'cpp'],
            'communication': ['communication', 'communicate', 'verbal', 'presentation'],
            'leadership': ['leadership', 'leader', 'lead', 'manage'],
            'teamwork': ['teamwork', 'team', 'collaborate', 'collaboration'],
            'problem_solving': ['problem-solving', 'problem solving', 'analytical', 'analytical'],
            'api': ['api', 'rest', 'graphql'],
            'devops': ['devops', 'docker', 'kubernetes', 'deployment'],
            'data_analysis': ['data', 'analysis', 'analytics', 'statistical'],
            'frontend': ['frontend', 'ui', 'ux', 'react', 'angular'],
            'backend': ['backend', 'server-side', 'api'],
        }
        
        for idx, row in self.metadata.iterrows():
            text = str(row['name']).lower() + " " + str(row['url']).lower()
            
            for skill, keywords in skill_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        if skill not in skill_index:
                            skill_index[skill] = set()
                        skill_index[skill].add(idx)
                        break
        
        return skill_index
    
    def extract_skills_and_requirements(self, query: str) -> Dict:
        """Advanced query analysis to extract skills and requirements"""
        query_lower = query.lower()
        
        extracted = {
            'technical_skills': [],
            'soft_skills': [],
            'experience_level': 'mid',
            'focus_areas': [],
            'keywords': []
        }
        
        # Technical skills extraction
        tech_mapping = {
            'python': ['python', 'py'],
            'java': ['java'],
            'sql': ['sql', 'database', 'database'],
            'javascript': ['javascript', 'js', 'node'],
            'c++': ['c++', 'cpp'],
            'react': ['react'],
            'angular': ['angular'],
            'aws': ['aws', 'amazon'],
            'azure': ['azure'],
            'docker': ['docker'],
            'kubernetes': ['kubernetes'],
            'api': ['api', 'rest', 'graphql']
        }
        
        for tech, keywords in tech_mapping.items():
            for kw in keywords:
                if kw in query_lower:
                    extracted['technical_skills'].append(tech)
                    break
        
        # Soft skills extraction
        soft_mapping = {
            'communication': ['communication', 'communicate', 'verbal', 'presentation'],
            'leadership': ['leadership', 'leader', 'lead', 'manage'],
            'teamwork': ['teamwork', 'team', 'collaborate', 'collaboration'],
            'problem-solving': ['problem-solving', 'problem solving', 'analytical'],
            'critical thinking': ['critical thinking', 'strategic']
        }
        
        for skill, keywords in soft_mapping.items():
            for kw in keywords:
                if kw in query_lower:
                    extracted['soft_skills'].append(skill)
                    break
        
        # Experience level detection
        if 'senior' in query_lower or '5+' in query_lower or '10+' in query_lower:
            extracted['experience_level'] = 'senior'
        elif 'junior' in query_lower or 'entry' in query_lower or 'graduate' in query_lower:
            extracted['experience_level'] = 'junior'
        
        # Extract focus areas
        if any(word in query_lower for word in ['frontend', 'ui', 'ux']):
            extracted['focus_areas'].append('frontend')
        if any(word in query_lower for word in ['backend', 'api', 'server']):
            extracted['focus_areas'].append('backend')
        if any(word in query_lower for word in ['devops', 'deployment', 'infrastructure']):
            extracted['focus_areas'].append('devops')
        if any(word in query_lower for word in ['data', 'analytics', 'ml', 'machine learning']):
            extracted['focus_areas'].append('data')
        
        return extracted
    
    def build_enhanced_query(self, query: str, extracted_info: Dict) -> str:
        """Build enhanced query with extracted information"""
        components = [query]
        
        # Add technical skills
        if extracted_info['technical_skills']:
            components.append(f"Technical: {' '.join(extracted_info['technical_skills'])}")
        
        # Add soft skills
        if extracted_info['soft_skills']:
            components.append(f"Soft skills: {' '.join(extracted_info['soft_skills'])}")
        
        # Add experience level
        components.append(f"Experience: {extracted_info['experience_level']}")
        
        # Add focus areas
        if extracted_info['focus_areas']:
            components.append(f"Focus: {' '.join(extracted_info['focus_areas'])}")
        
        return " ".join(components)
    
    def apply_skill_matching_boost(self, distances: np.ndarray, indices: np.ndarray, 
                                   extracted_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Boost scores based on skill matching"""
        boosted_distances = distances.copy()
        
        # Collect indices of assessments matching the extracted skills
        matching_indices = set()
        
        for skill in extracted_info['technical_skills'] + extracted_info['soft_skills']:
            if skill in self.skill_index:
                matching_indices.update(self.skill_index[skill])
        
        # Boost scores for matching assessments
        for i, idx in enumerate(indices[0]):
            if idx in matching_indices:
                boosted_distances[0][i] *= 1.3  # 30% boost for skill matches
        
        return boosted_distances, indices
    
    def diversity_reranking(self, candidates: List[Dict], k: int) -> List[Dict]:
        """Rerank to ensure diversity in assessment types"""
        if len(candidates) <= k:
            return candidates
        
        selected = []
        type_counts = {}
        
        # First pass: select highest scoring from each type
        for candidate in candidates:
            test_type = candidate.get('test_type', 'Unknown')
            
            if test_type not in type_counts:
                type_counts[test_type] = 0
            
            # Allow at most 40% from one type
            if type_counts[test_type] < k * 0.4:
                selected.append(candidate)
                type_counts[test_type] += 1
                
                if len(selected) >= k:
                    break
        
        # Fill remaining slots with highest scores
        if len(selected) < k:
            remaining = [c for c in candidates if c not in selected]
            selected.extend(remaining[:k - len(selected)])
        
        return selected[:k]
    
    def get_recommendations(self, query: str, k: int = 10) -> List[Dict]:
        """
        Get top-k recommendations with advanced ranking
        
        Args:
            query: Natural language query or job description
            k: Number of recommendations (5-10)
        
        Returns:
            List of assessment recommendations
        """
        
        try:
            # Validate k
            k = max(5, min(10, k))
            
            # Step 1: Extract skills and requirements
            extracted_info = self.extract_skills_and_requirements(query)
            
            # Step 2: Build enhanced query
            enhanced_query = self.build_enhanced_query(query, extracted_info)
            
            # Step 3: Generate embeddings for both original and enhanced queries
            original_embedding = self.model.encode([query])
            enhanced_embedding = self.model.encode([enhanced_query])
            
            # Combine embeddings (weighted average)
            combined_embedding = 0.6 * original_embedding + 0.4 * enhanced_embedding
            
            # Normalize
            faiss.normalize_L2(combined_embedding)
            
            # Step 4: Search FAISS index
            search_k = min(k * 3, len(self.metadata))
            distances, indices = self.index.search(combined_embedding, search_k)
            
            # Step 5: Apply skill matching boost
            distances, indices = self.apply_skill_matching_boost(distances, indices, extracted_info)
            
            # Step 6: Re-sort by boosted distances
            sorted_idx = np.argsort(-distances[0])
            distances = distances[:, sorted_idx]
            indices = indices[:, sorted_idx]
            
            # Step 7: Get candidate assessments
            candidates = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    assessment = self.metadata.iloc[idx].to_dict()
                    
                    # Normalize score to 0-1 range
                    normalized_score = float(min(score / 1.5, 1.0))
                    
                    candidates.append({
                        'name': assessment['name'],
                        'url': assessment['url'],
                        'test_type': str(assessment.get('test_type', 'Unknown')),
                        'duration': int(assessment.get('duration', 0)) if assessment.get('duration') else 0,
                        'relevance_score': normalized_score
                    })
            
            # Step 8: Apply diversity reranking
            final_recommendations = self.diversity_reranking(candidates, k)
            
            return final_recommendations
            
        except Exception as e:
            print(f"❌ Error in get_recommendations: {e}")
            return []

if __name__ == "__main__":
    recommender = AdvancedAssessmentRecommender()
    
    test_queries = [
        "I need a Java developer who can collaborate with business teams",
        "Looking for someone with Python skills and strong communication abilities",
        "Senior SQL database architect with 5+ years experience"
    ]
    
    for test_query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {test_query}")
        print('='*70)
        results = recommender.get_recommendations(test_query, k=10)
        for i, rec in enumerate(results, 1):
            print(f"{i}. {rec['name']} - Match: {rec['relevance_score']:.1%}")
