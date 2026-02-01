"""
Feature engineering for ML models
Converts student profiles into feature vectors
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from src.data_models import StudentProfile, StreamType, PersonalityType
from src.career_guidance_features import CareerGuidanceFeatureExtractor


class FeatureEngineer:
    """Feature engineering for student profiles"""
    
    def __init__(self, use_career_guidance: bool = True):
        """
        Initialize feature engineer
        
        Args:
            use_career_guidance: If True, include career guidance features
        """
        self.personality_encoding = self._create_personality_encoding()
        self.use_career_guidance = use_career_guidance
        self.career_guidance_extractor = None
        
        if use_career_guidance:
            try:
                self.career_guidance_extractor = CareerGuidanceFeatureExtractor()
                print("✅ Career guidance features enabled")
            except Exception as e:
                print(f"⚠️ Career guidance features disabled: {e}")
                self.use_career_guidance = False
    
    def _create_personality_encoding(self) -> Dict[PersonalityType, np.ndarray]:
        """Create one-hot encoding for personality types"""
        encoding = {}
        personality_types = list(PersonalityType)
        
        for i, ptype in enumerate(personality_types):
            encoding[ptype] = np.zeros(len(personality_types))
            encoding[ptype][i] = 1.0
        
        return encoding
    
    def extract_features(self, profile: StudentProfile) -> np.ndarray:
        """Extract feature vector from student profile"""
        features = []
        
        # Academic features (6 features)
        academic = profile.academic
        features.extend([
            academic.mathematics,
            academic.biology,
            academic.physics,
            academic.chemistry,
            academic.computer,
            academic.aggregate
        ])
        
        # Academic derived features
        features.append((academic.physics + academic.chemistry) / 2)  # Science average
        features.append((academic.mathematics + academic.physics) / 2)  # Math-Physics combo
        features.append((academic.biology + academic.chemistry) / 2)  # Bio-Chem combo
        
        # Aptitude features (5 features)
        aptitude = profile.aptitude
        features.extend([
            aptitude.mathematical_ability,
            aptitude.scientific_aptitude,
            aptitude.verbal_ability,
            aptitude.logical_reasoning,
            aptitude.spatial_ability
        ])
        
        # Aptitude derived features
        features.append(np.mean([
            aptitude.mathematical_ability,
            aptitude.logical_reasoning
        ]))  # Math-logic combo
        
        features.append(np.mean([
            aptitude.scientific_aptitude,
            aptitude.verbal_ability
        ]))  # Science-verbal combo
        
        # Interest features (5 features)
        interests = profile.interests
        features.extend([
            interests.medicine_healthcare,
            interests.engineering_technology,
            interests.computers_programming,
            interests.research_science,
            interests.creative_arts
        ])
        
        # Interest derived features
        features.append(interests.medicine_healthcare + interests.research_science)  # Medical interest
        features.append(interests.engineering_technology + interests.computers_programming)  # Tech interest
        
        # Personality encoding (16 features for 16 MBTI types)
        personality_vec = self.personality_encoding[profile.personality]
        features.extend(personality_vec.tolist())
        
        # Additional interaction features
        features.append(academic.mathematics * aptitude.mathematical_ability / 100)  # Math alignment
        features.append(academic.biology * interests.medicine_healthcare / 100)  # Medical alignment
        features.append(academic.computer * interests.computers_programming / 100)  # CS alignment
        
        # Career guidance features (if enabled)
        if self.use_career_guidance and self.career_guidance_extractor:
            try:
                cg_features = self.career_guidance_extractor.extract_career_guidance_features(profile)
                features.extend(cg_features.tolist())
            except Exception as e:
                # If career guidance fails, add zeros
                cg_count = self.career_guidance_extractor.get_feature_count()
                features.extend([0.0] * cg_count)
        
        return np.array(features, dtype=np.float32)
    
    def extract_features_batch(self, profiles: List[StudentProfile]) -> np.ndarray:
        """Extract features for multiple profiles"""
        return np.array([self.extract_features(profile) for profile in profiles])
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        names = []
        
        # Academic
        names.extend(['math', 'bio', 'physics', 'chem', 'computer', 'aggregate',
                     'science_avg', 'math_physics_avg', 'bio_chem_avg'])
        
        # Aptitude
        names.extend(['apt_math', 'apt_science', 'apt_verbal', 'apt_logical', 'apt_spatial',
                     'apt_math_logic_avg', 'apt_science_verbal_avg'])
        
        # Interests
        names.extend(['int_medical', 'int_engineering', 'int_computer', 'int_research', 'int_arts',
                     'int_medical_combo', 'int_tech_combo'])
        
        # Personality (one-hot)
        names.extend([f'personality_{ptype.value}' for ptype in PersonalityType])
        
        # Interactions
        names.extend(['math_apt_alignment', 'bio_medical_alignment', 'cs_interest_alignment'])
        
        # Career guidance features (if enabled)
        if self.use_career_guidance and self.career_guidance_extractor:
            try:
                cg_names = self.career_guidance_extractor.get_feature_names()
                names.extend(cg_names)
            except Exception:
                # Add placeholder names if extractor not available
                cg_count = 17  # Default count
                names.extend([f'cg_feature_{i}' for i in range(cg_count)])
        
        return names
    
    def create_dataframe(self, profiles: List[StudentProfile]) -> pd.DataFrame:
        """Create pandas DataFrame from profiles"""
        features = self.extract_features_batch(profiles)
        feature_names = self.get_feature_names()
        
        return pd.DataFrame(features, columns=feature_names)
    
    def extract_target_labels(self, profiles: List[StudentProfile], 
                            recommendations: List[StreamType]) -> np.ndarray:
        """Extract target labels for classification"""
        stream_to_int = {
            StreamType.PRE_MEDICAL: 0,
            StreamType.PRE_ENGINEERING: 1,
            StreamType.ICS: 2,
            StreamType.GENERAL_SCIENCE: 3
        }
        
        return np.array([stream_to_int[rec] for rec in recommendations])
    
    def extract_target_scores(self, profiles: List[StudentProfile],
                            scorer) -> np.ndarray:
        """Extract target scores for each stream (for regression/ranking)"""
        scores = []
        
        for profile in profiles:
            stream_scores = []
            for stream in StreamType:
                result = scorer.calculate_final_score(profile, stream)
                stream_scores.append(result['final_score'])
            scores.append(stream_scores)
        
        return np.array(scores, dtype=np.float32)
