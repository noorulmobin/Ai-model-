"""
Prediction interface for FSC Recommendation System
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List

from src.data_models import StudentProfile, StreamType
from src.feature_engineer import FeatureEngineer
from src.scoring import RuleBasedScorer
from src.career_guidance import CareerGuidanceQA, integrate_with_recommendation


class FSCPredictor:
    """Predictor class for making recommendations"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None, 
                 use_career_guidance: bool = True):
        self.feature_engineer = FeatureEngineer()
        self.scorer = RuleBasedScorer()
        self.model = None
        self.scaler = None
        self.career_qa = None
        
        if use_career_guidance:
            try:
                self.career_qa = CareerGuidanceQA()
            except Exception as e:
                print(f"⚠️ Career guidance not available: {e}")
                self.career_qa = None
        
        if model_path and scaler_path:
            self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str):
        """Load trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
    
    def predict_stream(self, profile: StudentProfile) -> StreamType:
        """Predict recommended stream using ML model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract features
        features = self.feature_engineer.extract_features(profile)
        features = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Map to stream type
        stream_map = {
            0: StreamType.PRE_MEDICAL,
            1: StreamType.PRE_ENGINEERING,
            2: StreamType.ICS,
            3: StreamType.GENERAL_SCIENCE
        }
        
        return stream_map[prediction]
    
    def predict_proba(self, profile: StudentProfile) -> Dict[StreamType, float]:
        """Get prediction probabilities for each stream"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract features
        features = self.feature_engineer.extract_features(profile)
        features = self.scaler.transform(features.reshape(1, -1))
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
        else:
            # For models without predict_proba, use decision function
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(features)[0]
                # Convert to probabilities (softmax)
                exp_decision = np.exp(decision - np.max(decision))
                proba = exp_decision / exp_decision.sum()
            else:
                raise ValueError("Model does not support probability prediction")
        
        stream_map = {
            0: StreamType.PRE_MEDICAL,
            1: StreamType.PRE_ENGINEERING,
            2: StreamType.ICS,
            3: StreamType.GENERAL_SCIENCE
        }
        
        return {stream_map[i]: float(prob) for i, prob in enumerate(proba)}
    
    def recommend(self, profile: StudentProfile, use_ml: bool = True) -> Dict:
        """
        Generate recommendation using ML model or rule-based system
        
        Args:
            profile: Student profile
            use_ml: If True, use ML model; if False, use rule-based system
        
        Returns:
            Dictionary with recommendation details
        """
        if use_ml and self.model is not None:
            # Use ML model
            predicted_stream = self.predict_stream(profile)
            probabilities = self.predict_proba(profile)
            
            # Get rule-based recommendation for comparison
            rule_based_result = self.scorer.recommend(profile)
            
            result = {
                'ml_prediction': predicted_stream.value,
                'ml_probabilities': {k.value: v for k, v in probabilities.items()},
                'rule_based_recommendation': rule_based_result.top_recommendation.stream.value,
                'rule_based_scores': {
                    rec.stream.value: rec.match_percentage 
                    for rec in rule_based_result.recommendations
                },
                'warnings': rule_based_result.warnings,
                'suggestions': rule_based_result.suggestions
            }
            
            # Add career guidance if available
            if self.career_qa:
                result = integrate_with_recommendation(result, self.career_qa)
            
            return result
        else:
            # Use rule-based system
            result_obj = self.scorer.recommend(profile)
            
            result = {
                'recommended_stream': result_obj.top_recommendation.stream.value,
                'match_percentage': result_obj.top_recommendation.match_percentage,
                'all_recommendations': [
                    {
                        'stream': rec.stream.value,
                        'match_percentage': rec.match_percentage,
                        'suitability': rec.suitability_class,
                        'academic_score': rec.academic_score,
                        'aptitude_score': rec.aptitude_score,
                        'personality_score': rec.personality_score,
                        'interest_score': rec.interest_score,
                        'reasoning': rec.reasoning
                    }
                    for rec in result_obj.recommendations
                ],
                'warnings': result_obj.warnings,
                'suggestions': result_obj.suggestions
            }
            
            # Add career guidance if available
            if self.career_qa:
                result = integrate_with_recommendation(result, self.career_qa)
            
            return result
