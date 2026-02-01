"""
Career Guidance Q&A Integration
Integrates Hugging Face career guidance dataset with FSC recommendations
"""
from typing import List, Dict, Optional
from datasets import load_dataset
import pandas as pd
from pathlib import Path


class CareerGuidanceQA:
    """Career guidance Q&A system using Hugging Face dataset"""
    
    def __init__(self, dataset_name: str = "Pradeep016/career-guidance-qa-dataset", 
                 cache_dir: str = "data/cache"):
        """
        Initialize career guidance Q&A system
        
        Args:
            dataset_name: Hugging Face dataset identifier
            cache_dir: Directory to cache the dataset
        """
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        self.df = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from Hugging Face"""
        try:
            print(f"Loading career guidance dataset: {self.dataset_name}...")
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Convert to pandas DataFrame for easier querying
            # Handle different dataset splits
            if 'train' in self.dataset:
                self.df = pd.DataFrame(self.dataset['train'])
            elif 'validation' in self.dataset:
                self.df = pd.DataFrame(self.dataset['validation'])
            else:
                # Use first available split
                split_name = list(self.dataset.keys())[0]
                self.df = pd.DataFrame(self.dataset[split_name])
            
            print(f"✅ Dataset loaded: {len(self.df)} Q&A pairs")
            print(f"Columns: {list(self.df.columns)}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load dataset: {e}")
            print("Career guidance features will be limited.")
            self.df = pd.DataFrame()
    
    def search_qa(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant Q&A pairs based on query
        
        Args:
            query: Search query/question
            top_k: Number of results to return
        
        Returns:
            List of relevant Q&A pairs
        """
        if self.df is None or len(self.df) == 0:
            return []
        
        query_lower = query.lower()
        results = []
        
        # Simple keyword-based search
        # Check if query matches question or answer
        for idx, row in self.df.iterrows():
            score = 0
            
            # Check question column (common names)
            for col in ['question', 'Question', 'input', 'Input', 'query', 'Query']:
                if col in row and pd.notna(row[col]):
                    if query_lower in str(row[col]).lower():
                        score += 2
            
            # Check answer column (common names)
            for col in ['answer', 'Answer', 'output', 'Output', 'response', 'Response']:
                if col in row and pd.notna(row[col]):
                    if query_lower in str(row[col]).lower():
                        score += 1
            
            if score > 0:
                # Extract question and answer
                qa_pair = {
                    'score': score,
                    'question': self._extract_field(row, ['question', 'Question', 'input', 'Input', 'query', 'Query']),
                    'answer': self._extract_field(row, ['answer', 'Answer', 'output', 'Output', 'response', 'Response']),
                    'raw': row.to_dict()
                }
                results.append(qa_pair)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _extract_field(self, row: pd.Series, possible_names: List[str]) -> str:
        """Extract field from row using possible column names"""
        for name in possible_names:
            if name in row and pd.notna(row[name]):
                return str(row[name])
        return ""
    
    def get_stream_guidance(self, stream: str) -> List[Dict]:
        """
        Get career guidance Q&A for a specific FSC stream
        
        Args:
            stream: Stream name (Pre-Medical, Pre-Engineering, ICS, General Science)
        
        Returns:
            List of relevant Q&A pairs
        """
        # Search terms for each stream
        search_terms = {
            'Pre-Medical': ['medical', 'doctor', 'medicine', 'healthcare', 'biology', 'pre-medical'],
            'Pre-Engineering': ['engineering', 'engineer', 'technical', 'pre-engineering', 'mechanical', 'electrical'],
            'ICS': ['computer', 'IT', 'software', 'programming', 'ICS', 'information technology'],
            'General Science': ['science', 'general science', 'bachelor', 'BS']
        }
        
        terms = search_terms.get(stream, [stream.lower()])
        all_results = []
        
        for term in terms:
            results = self.search_qa(term, top_k=3)
            all_results.extend(results)
        
        # Remove duplicates and sort by score
        seen = set()
        unique_results = []
        for result in all_results:
            key = (result['question'], result['answer'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results[:5]
    
    def get_career_advice(self, topic: str) -> List[Dict]:
        """
        Get career advice for a specific topic
        
        Args:
            topic: Career topic (e.g., 'career choice', 'job market', 'skills')
        
        Returns:
            List of relevant Q&A pairs
        """
        return self.search_qa(topic, top_k=5)
    
    def get_all_qa(self) -> pd.DataFrame:
        """Get all Q&A pairs as DataFrame"""
        return self.df.copy() if self.df is not None else pd.DataFrame()


def integrate_with_recommendation(recommendation_result: Dict, 
                                 career_qa: CareerGuidanceQA) -> Dict:
    """
    Integrate career guidance Q&A with recommendation results
    
    Args:
        recommendation_result: Result from FSCPredictor.recommend()
        career_qa: CareerGuidanceQA instance
    
    Returns:
        Enhanced recommendation result with career guidance
    """
    enhanced_result = recommendation_result.copy()
    
    # Get recommended stream
    recommended_stream = None
    if 'ml_prediction' in recommendation_result:
        recommended_stream = recommendation_result['ml_prediction']
    elif 'recommended_stream' in recommendation_result:
        recommended_stream = recommendation_result['recommended_stream']
    
    # Add career guidance Q&A
    if recommended_stream:
        guidance = career_qa.get_stream_guidance(recommended_stream)
        enhanced_result['career_guidance'] = guidance
    
    # Add general career advice
    general_advice = career_qa.get_career_advice('career guidance')
    enhanced_result['general_career_advice'] = general_advice
    
    return enhanced_result
