"""
Data models and schemas for FSC Recommendation System
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum


class StreamType(Enum):
    """FSC Stream types"""
    PRE_MEDICAL = "Pre-Medical"
    PRE_ENGINEERING = "Pre-Engineering"
    ICS = "ICS"
    GENERAL_SCIENCE = "General Science"


class PersonalityType(Enum):
    """MBTI Personality Types"""
    INTP = "INTP"
    INTJ = "INTJ"
    ISFJ = "ISFJ"
    ISFP = "ISFP"
    ISTP = "ISTP"
    ISTJ = "ISTJ"
    INFP = "INFP"
    INFJ = "INFJ"
    ENFP = "ENFP"
    ENFJ = "ENFJ"
    ENTP = "ENTP"
    ENTJ = "ENTJ"
    ESFP = "ESFP"
    ESFJ = "ESFJ"
    ESTP = "ESTP"
    ESTJ = "ESTJ"


@dataclass
class AcademicPerformance:
    """Matriculation academic performance"""
    mathematics: float  # 0-100
    biology: float  # 0-100
    physics: float  # 0-100
    chemistry: float  # 0-100
    computer: float  # 0-100
    aggregate: float  # 0-100
    
    def validate(self):
        """Validate marks are in valid range"""
        for mark in [self.mathematics, self.biology, self.physics, 
                    self.chemistry, self.computer, self.aggregate]:
            if not 0 <= mark <= 100:
                raise ValueError(f"Mark {mark} must be between 0 and 100")


@dataclass
class AptitudeScores:
    """Aptitude test scores"""
    mathematical_ability: float  # 0-100
    scientific_aptitude: float  # 0-100
    verbal_ability: float  # 0-100
    logical_reasoning: float  # 0-100
    spatial_ability: float  # 0-100
    
    def validate(self):
        """Validate scores are in valid range"""
        for score in [self.mathematical_ability, self.scientific_aptitude,
                     self.verbal_ability, self.logical_reasoning, 
                     self.spatial_ability]:
            if not 0 <= score <= 100:
                raise ValueError(f"Score {score} must be between 0 and 100")


@dataclass
class InterestScores:
    """Interest assessment scores"""
    medicine_healthcare: float  # 0-100
    engineering_technology: float  # 0-100
    computers_programming: float  # 0-100
    research_science: float  # 0-100
    creative_arts: float  # 0-100
    
    def validate(self):
        """Validate interest scores are in valid range"""
        for score in [self.medicine_healthcare, self.engineering_technology,
                     self.computers_programming, self.research_science,
                     self.creative_arts]:
            if not 0 <= score <= 100:
                raise ValueError(f"Interest score {score} must be between 0 and 100")


@dataclass
class StudentProfile:
    """Complete student profile"""
    academic: AcademicPerformance
    aptitude: AptitudeScores
    personality: PersonalityType
    interests: InterestScores
    student_id: Optional[str] = None
    
    def validate(self):
        """Validate all components"""
        self.academic.validate()
        self.aptitude.validate()
        self.interests.validate()


@dataclass
class StreamRecommendation:
    """Recommendation for a specific stream"""
    stream: StreamType
    match_percentage: float  # 0-100
    academic_score: float
    aptitude_score: float
    personality_score: float
    interest_score: float
    suitability_class: str  # "Highly Recommended", "Recommended", etc.
    reasoning: List[str]  # Explanation points


@dataclass
class RecommendationResult:
    """Complete recommendation result"""
    student_id: Optional[str]
    recommendations: List[StreamRecommendation]
    top_recommendation: StreamRecommendation
    warnings: List[str]  # Red flags or concerns
    suggestions: List[str]  # Improvement suggestions
