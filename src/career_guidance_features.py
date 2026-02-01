"""
Career Guidance Feature Extractor
Extracts features from career guidance dataset to enhance training
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from src.data_models import StudentProfile, StreamType
from src.career_guidance import CareerGuidanceQA


class CareerGuidanceFeatureExtractor:
    """Extract features from career guidance dataset for training"""
    
    def __init__(self, career_qa: Optional[CareerGuidanceQA] = None):
        """
        Initialize feature extractor
        
        Args:
            career_qa: CareerGuidanceQA instance (will create if None)
        """
        if career_qa is None:
            try:
                self.career_qa = CareerGuidanceQA()
            except Exception as e:
                print(f"⚠️ Warning: Could not load career guidance dataset: {e}")
                self.career_qa = None
        else:
            self.career_qa = career_qa
        
        # Pre-compute stream keywords for faster matching
        self.stream_keywords = self._initialize_stream_keywords()
    
    def _initialize_stream_keywords(self) -> Dict[StreamType, List[str]]:
        """Initialize keywords for each stream"""
        return {
            StreamType.PRE_MEDICAL: [
                'medical', 'doctor', 'medicine', 'healthcare', 'biology', 
                'pre-medical', 'hospital', 'patient', 'treatment', 'diagnosis',
                'nurse', 'dentist', 'pharmacy', 'physiotherapy'
            ],
            StreamType.PRE_ENGINEERING: [
                'engineering', 'engineer', 'technical', 'pre-engineering',
                'mechanical', 'electrical', 'civil', 'chemical', 'software',
                'construction', 'design', 'technology', 'innovation'
            ],
            StreamType.ICS: [
                'computer', 'IT', 'software', 'programming', 'ICS',
                'information technology', 'coding', 'developer', 'technology',
                'software engineering', 'data science', 'cyber security',
                'web development', 'artificial intelligence'
            ],
            StreamType.GENERAL_SCIENCE: [
                'science', 'general science', 'bachelor', 'BS', 'research',
                'academic', 'education', 'arts', 'humanities', 'social science'
            ]
        }
    
    def extract_career_guidance_features(self, profile: StudentProfile) -> np.ndarray:
        """
        Extract career guidance-based features from student profile
        
        Args:
            profile: Student profile
        
        Returns:
            Array of career guidance features
        """
        if self.career_qa is None or self.career_qa.df is None or len(self.career_qa.df) == 0:
            # Return zero features if dataset not available
            return np.zeros(self.get_feature_count(), dtype=np.float32)
        
        features = []
        
        # Feature 1-4: Stream-specific career guidance relevance scores
        for stream in StreamType:
            relevance_score = self._calculate_stream_relevance(profile, stream)
            features.append(relevance_score)
        
        # Feature 5-8: Career guidance alignment with student interests
        interest_alignment = self._calculate_interest_alignment(profile)
        features.extend(interest_alignment)
        
        # Feature 9-12: Career guidance alignment with academic performance
        academic_alignment = self._calculate_academic_alignment(profile)
        features.extend(academic_alignment)
        
        # Feature 13-16: Career guidance alignment with aptitude
        aptitude_alignment = self._calculate_aptitude_alignment(profile)
        features.extend(aptitude_alignment)
        
        # Feature 17: Overall career guidance match score
        overall_match = self._calculate_overall_match(profile)
        features.append(overall_match)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_stream_relevance(self, profile: StudentProfile, stream: StreamType) -> float:
        """Calculate how relevant career guidance is for this stream"""
        if self.career_qa is None:
            return 0.0
        
        # Get guidance for this stream
        guidance = self.career_qa.get_stream_guidance(stream.value)
        
        if not guidance:
            return 0.0
        
        # Calculate relevance based on:
        # 1. Number of relevant Q&A pairs found
        # 2. Average score of matches
        # 3. Alignment with student profile
        
        relevance = 0.0
        
        # Base relevance from number of matches
        relevance += min(len(guidance) / 5.0, 1.0) * 30.0
        
        # Average score of matches
        if guidance:
            avg_score = np.mean([qa['score'] for qa in guidance])
            relevance += (avg_score / 10.0) * 20.0
        
        # Profile alignment (check if student interests/aptitude match stream keywords)
        keywords = self.stream_keywords.get(stream, [])
        profile_text = self._profile_to_text(profile).lower()
        
        keyword_matches = sum(1 for keyword in keywords if keyword in profile_text)
        if keywords:
            keyword_score = keyword_matches / len(keywords)
            relevance += keyword_score * 50.0
        
        return min(relevance, 100.0)  # Cap at 100
    
    def _calculate_interest_alignment(self, profile: StudentProfile) -> List[float]:
        """Calculate alignment between interests and career guidance"""
        alignments = []
        
        interest_mapping = {
            0: ('medicine_healthcare', StreamType.PRE_MEDICAL),
            1: ('engineering_technology', StreamType.PRE_ENGINEERING),
            2: ('computers_programming', StreamType.ICS),
            3: ('research_science', StreamType.GENERAL_SCIENCE)
        }
        
        for idx, (interest_name, stream) in interest_mapping.items():
            interest_value = getattr(profile.interests, interest_name, 0)
            
            # Get career guidance for this stream
            if self.career_qa:
                guidance = self.career_qa.get_stream_guidance(stream.value)
                guidance_score = len(guidance) * 10.0 if guidance else 0.0
            else:
                guidance_score = 0.0
            
            # Combine interest value with guidance relevance
            alignment = (interest_value * 0.6) + (guidance_score * 0.4)
            alignments.append(alignment)
        
        return alignments
    
    def _calculate_academic_alignment(self, profile: StudentProfile) -> List[float]:
        """Calculate alignment between academic performance and career guidance"""
        alignments = []
        
        academic_mapping = {
            0: ('biology', StreamType.PRE_MEDICAL),
            1: ('mathematics', StreamType.PRE_ENGINEERING),
            2: ('computer', StreamType.ICS),
            3: ('aggregate', StreamType.GENERAL_SCIENCE)
        }
        
        for idx, (subject_name, stream) in academic_mapping.items():
            academic_value = getattr(profile.academic, subject_name, 0)
            
            # Get career guidance relevance
            if self.career_qa:
                guidance = self.career_qa.get_stream_guidance(stream.value)
                guidance_score = len(guidance) * 10.0 if guidance else 0.0
            else:
                guidance_score = 0.0
            
            # Combine academic performance with guidance
            alignment = (academic_value * 0.7) + (guidance_score * 0.3)
            alignments.append(alignment)
        
        return alignments
    
    def _calculate_aptitude_alignment(self, profile: StudentProfile) -> List[float]:
        """Calculate alignment between aptitude and career guidance"""
        alignments = []
        
        # Map aptitudes to streams
        aptitude_stream_mapping = {
            0: ('mathematical_ability', StreamType.PRE_ENGINEERING),
            1: ('scientific_aptitude', StreamType.PRE_MEDICAL),
            2: ('logical_reasoning', StreamType.ICS),
            3: ('verbal_ability', StreamType.GENERAL_SCIENCE)
        }
        
        for idx, (aptitude_name, stream) in aptitude_stream_mapping.items():
            aptitude_value = getattr(profile.aptitude, aptitude_name, 0)
            
            # Get career guidance relevance
            if self.career_qa:
                guidance = self.career_qa.get_stream_guidance(stream.value)
                guidance_score = len(guidance) * 10.0 if guidance else 0.0
            else:
                guidance_score = 0.0
            
            # Combine aptitude with guidance
            alignment = (aptitude_value * 0.7) + (guidance_score * 0.3)
            alignments.append(alignment)
        
        return alignments
    
    def _calculate_overall_match(self, profile: StudentProfile) -> float:
        """Calculate overall career guidance match score"""
        if self.career_qa is None:
            return 0.0
        
        # Get guidance for all streams
        total_guidance = 0
        for stream in StreamType:
            guidance = self.career_qa.get_stream_guidance(stream.value)
            total_guidance += len(guidance)
        
        # Normalize
        if total_guidance > 0:
            return min(total_guidance / 20.0 * 100.0, 100.0)
        return 0.0
    
    def _profile_to_text(self, profile: StudentProfile) -> str:
        """Convert profile to text for keyword matching"""
        text_parts = []
        
        # Add interests
        text_parts.append(f"medical interest: {profile.interests.medicine_healthcare}")
        text_parts.append(f"engineering interest: {profile.interests.engineering_technology}")
        text_parts.append(f"computer interest: {profile.interests.computers_programming}")
        
        # Add academic strengths
        if profile.academic.biology > 70:
            text_parts.append("biology")
        if profile.academic.mathematics > 70:
            text_parts.append("mathematics engineering")
        if profile.academic.computer > 70:
            text_parts.append("computer technology")
        
        return " ".join(text_parts)
    
    def get_feature_count(self) -> int:
        """Get number of career guidance features"""
        return 17  # 4 stream relevance + 4 interest + 4 academic + 4 aptitude + 1 overall
    
    def get_feature_names(self) -> List[str]:
        """Get names of career guidance features"""
        names = []
        
        # Stream relevance
        names.extend([f'cg_relevance_{stream.value.lower().replace("-", "_")}' 
                     for stream in StreamType])
        
        # Interest alignment
        names.extend(['cg_interest_medical', 'cg_interest_engineering', 
                     'cg_interest_computer', 'cg_interest_research'])
        
        # Academic alignment
        names.extend(['cg_academic_medical', 'cg_academic_engineering',
                     'cg_academic_computer', 'cg_academic_general'])
        
        # Aptitude alignment
        names.extend(['cg_aptitude_engineering', 'cg_aptitude_medical',
                     'cg_aptitude_computer', 'cg_aptitude_general'])
        
        # Overall
        names.append('cg_overall_match')
        
        return names
