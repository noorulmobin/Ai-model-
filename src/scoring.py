"""
Rule-based scoring system implementing the documented algorithm
"""
from typing import Dict, List
from src.data_models import (
    StudentProfile, StreamRecommendation, StreamType, 
    PersonalityType, RecommendationResult
)


class RuleBasedScorer:
    """Implements the rule-based scoring algorithm from documentation"""
    
    # Personality match scores (MBTI -> Stream compatibility)
    PERSONALITY_MATCHES = {
        PersonalityType.INTP: {
            StreamType.PRE_ENGINEERING: 90,
            StreamType.ICS: 95,
            StreamType.PRE_MEDICAL: 60,
            StreamType.GENERAL_SCIENCE: 70
        },
        PersonalityType.INTJ: {
            StreamType.PRE_ENGINEERING: 95,
            StreamType.ICS: 90,
            StreamType.PRE_MEDICAL: 70,
            StreamType.GENERAL_SCIENCE: 75
        },
        PersonalityType.ISFJ: {
            StreamType.PRE_MEDICAL: 95,
            StreamType.GENERAL_SCIENCE: 80,
            StreamType.PRE_ENGINEERING: 50,
            StreamType.ICS: 60
        },
        PersonalityType.ENFP: {
            StreamType.PRE_MEDICAL: 80,
            StreamType.GENERAL_SCIENCE: 85,
            StreamType.ICS: 70,
            StreamType.PRE_ENGINEERING: 65
        },
        # Add more personality types as needed
        # Default fallback for unlisted types
    }
    
    def __init__(self):
        # Set default personality matches for unlisted types
        self._set_default_personality_matches()
    
    def _set_default_personality_matches(self):
        """Set default personality matches for unlisted MBTI types"""
        default_matches = {
            StreamType.PRE_ENGINEERING: 70,
            StreamType.ICS: 75,
            StreamType.PRE_MEDICAL: 70,
            StreamType.GENERAL_SCIENCE: 75
        }
        
        for ptype in PersonalityType:
            if ptype not in self.PERSONALITY_MATCHES:
                self.PERSONALITY_MATCHES[ptype] = default_matches.copy()
    
    def calculate_academic_score(self, profile: StudentProfile, stream: StreamType) -> float:
        """Calculate academic performance score for a stream"""
        academic = profile.academic
        score = 0.0
        
        if stream == StreamType.PRE_MEDICAL:
            # Biology marks (≥75% = 25 points, 60-74% = 15 points)
            if academic.biology >= 75:
                score += 25
            elif academic.biology >= 60:
                score += 15
            
            # Science average (Physics + Chemistry) ≥75% = 20 points
            science_avg = (academic.physics + academic.chemistry) / 2
            if science_avg >= 75:
                score += 20
            elif science_avg >= 65:
                score += 15
        
        elif stream == StreamType.PRE_ENGINEERING:
            # Mathematics marks (≥80% = 30 points, 65-79% = 20 points)
            if academic.mathematics >= 80:
                score += 30
            elif academic.mathematics >= 65:
                score += 20
            
            # Science average ≥75% = 20 points
            science_avg = (academic.physics + academic.chemistry) / 2
            if science_avg >= 75:
                score += 20
            elif science_avg >= 65:
                score += 15
        
        elif stream == StreamType.ICS:
            # Mathematics marks ≥70% = 20 points
            if academic.mathematics >= 70:
                score += 20
            elif academic.mathematics >= 60:
                score += 15
            
            # Computer marks ≥70% = 20 points
            if academic.computer >= 70:
                score += 20
            elif academic.computer >= 60:
                score += 15
        
        elif stream == StreamType.GENERAL_SCIENCE:
            # More flexible scoring
            avg_score = (academic.mathematics + academic.physics + 
                        academic.chemistry + academic.biology) / 4
            if avg_score >= 70:
                score += 30
            elif avg_score >= 60:
                score += 20
            elif avg_score >= 50:
                score += 15
        
        return min(score, 50.0)  # Cap at 50 points (out of 100 total)
    
    def calculate_aptitude_score(self, profile: StudentProfile, stream: StreamType) -> float:
        """Calculate aptitude test score for a stream"""
        aptitude = profile.aptitude
        score = 0.0
        
        if stream == StreamType.PRE_MEDICAL:
            # Scientific Aptitude ≥75% = 35 points
            if aptitude.scientific_aptitude >= 75:
                score += 35
            elif aptitude.scientific_aptitude >= 60:
                score += 25
            
            # Verbal Ability ≥70% = 15 points
            if aptitude.verbal_ability >= 70:
                score += 15
            elif aptitude.verbal_ability >= 60:
                score += 10
        
        elif stream == StreamType.PRE_ENGINEERING:
            # Mathematical Ability ≥80% = 40 points
            if aptitude.mathematical_ability >= 80:
                score += 40
            elif aptitude.mathematical_ability >= 70:
                score += 30
            
            # Scientific Aptitude ≥75% = 30 points
            if aptitude.scientific_aptitude >= 75:
                score += 30
            elif aptitude.scientific_aptitude >= 60:
                score += 20
            
            # Logical Reasoning (bonus)
            if aptitude.logical_reasoning >= 70:
                score += 12
        
        elif stream == StreamType.ICS:
            # Mathematical Ability ≥80% = 35 points
            if aptitude.mathematical_ability >= 80:
                score += 35
            elif aptitude.mathematical_ability >= 70:
                score += 25
            
            # Logical Reasoning ≥70% = 30 points
            if aptitude.logical_reasoning >= 70:
                score += 30
            elif aptitude.logical_reasoning >= 60:
                score += 20
        
        elif stream == StreamType.GENERAL_SCIENCE:
            # Balanced scoring
            avg_aptitude = (aptitude.mathematical_ability + 
                          aptitude.scientific_aptitude + 
                          aptitude.verbal_ability) / 3
            if avg_aptitude >= 70:
                score += 40
            elif avg_aptitude >= 60:
                score += 30
        
        return min(score, 50.0)  # Cap at 50 points
    
    def calculate_personality_score(self, profile: StudentProfile, stream: StreamType) -> float:
        """Calculate personality match score"""
        personality = profile.personality
        
        if personality in self.PERSONALITY_MATCHES:
            return self.PERSONALITY_MATCHES[personality].get(stream, 70)
        return 70  # Default
    
    def calculate_interest_score(self, profile: StudentProfile, stream: StreamType) -> float:
        """Calculate interest alignment score"""
        interests = profile.interests
        score = 0.0
        
        if stream == StreamType.PRE_MEDICAL:
            # Biology/Medicine/Healthcare: +30 points
            if interests.medicine_healthcare >= 80:
                score += 30
            elif interests.medicine_healthcare >= 60:
                score += 20
            
            # Chemistry/Science: +20 points
            if interests.research_science >= 70:
                score += 20
        
        elif stream == StreamType.PRE_ENGINEERING:
            # Mathematics/Physics/Engineering: +30 points
            if interests.engineering_technology >= 80:
                score += 30
            elif interests.engineering_technology >= 60:
                score += 20
            
            # Chemistry/Science: +20 points
            if interests.research_science >= 70:
                score += 20
        
        elif stream == StreamType.ICS:
            # Computers/Programming/Technology: +40 points
            if interests.computers_programming >= 80:
                score += 40
            elif interests.computers_programming >= 60:
                score += 30
            
            # Mathematics/Engineering: +20 points
            if interests.engineering_technology >= 70:
                score += 20
        
        elif stream == StreamType.GENERAL_SCIENCE:
            # Arts/Design: +30 points
            if interests.creative_arts >= 70:
                score += 30
            elif interests.creative_arts >= 50:
                score += 20
        
        return min(score, 50.0)  # Cap at 50 points
    
    def calculate_final_score(self, profile: StudentProfile, stream: StreamType) -> Dict:
        """Calculate final recommendation score for a stream"""
        academic_score = self.calculate_academic_score(profile, stream)
        aptitude_score = self.calculate_aptitude_score(profile, stream)
        personality_score = self.calculate_personality_score(profile, stream)
        interest_score = self.calculate_interest_score(profile, stream)
        
        # Weighted final score
        final_score = (
            academic_score * 0.40 +
            aptitude_score * 0.35 +
            personality_score * 0.15 +
            interest_score * 0.10
        )
        
        return {
            'academic_score': academic_score,
            'aptitude_score': aptitude_score,
            'personality_score': personality_score,
            'interest_score': interest_score,
            'final_score': final_score
        }
    
    def get_suitability_class(self, match_percentage: float) -> str:
        """Get suitability classification"""
        if match_percentage >= 80:
            return "Highly Recommended"
        elif match_percentage >= 65:
            return "Recommended"
        elif match_percentage >= 50:
            return "Moderately Suitable"
        elif match_percentage >= 40:
            return "Marginally Suitable"
        else:
            return "Not Recommended"
    
    def check_red_flags(self, profile: StudentProfile) -> List[str]:
        """Check for red flags that should prevent certain recommendations"""
        warnings = []
        academic = profile.academic
        aptitude = profile.aptitude
        
        # Pre-Medical red flags
        if academic.biology < 60:
            warnings.append("Biology marks below 60% - Not suitable for Pre-Medical")
        if academic.chemistry < 55:
            warnings.append("Chemistry marks below 55% - Not suitable for Pre-Medical")
        
        # Pre-Engineering red flags
        if academic.mathematics < 65:
            warnings.append("Mathematics marks below 65% - Not suitable for Pre-Engineering")
        if academic.physics < 60:
            warnings.append("Physics marks below 60% - Not suitable for Pre-Engineering")
        
        # ICS red flags
        if academic.computer < 50:
            warnings.append("Computer marks below 50% - Not suitable for ICS")
        if aptitude.logical_reasoning < 60:
            warnings.append("Logical reasoning below 60% - May struggle with ICS")
        
        return warnings
    
    def recommend(self, profile: StudentProfile) -> RecommendationResult:
        """Generate recommendations for a student"""
        profile.validate()
        
        recommendations = []
        warnings = self.check_red_flags(profile)
        
        for stream in StreamType:
            scores = self.calculate_final_score(profile, stream)
            
            recommendation = StreamRecommendation(
                stream=stream,
                match_percentage=scores['final_score'],
                academic_score=scores['academic_score'],
                aptitude_score=scores['aptitude_score'],
                personality_score=scores['personality_score'],
                interest_score=scores['interest_score'],
                suitability_class=self.get_suitability_class(scores['final_score']),
                reasoning=self._generate_reasoning(profile, stream, scores)
            )
            recommendations.append(recommendation)
        
        # Sort by match percentage
        recommendations.sort(key=lambda x: x.match_percentage, reverse=True)
        top_recommendation = recommendations[0]
        
        return RecommendationResult(
            student_id=profile.student_id,
            recommendations=recommendations,
            top_recommendation=top_recommendation,
            warnings=warnings,
            suggestions=self._generate_suggestions(profile, top_recommendation)
        )
    
    def _generate_reasoning(self, profile: StudentProfile, stream: StreamType, 
                           scores: Dict) -> List[str]:
        """Generate reasoning points for recommendation"""
        reasoning = []
        
        if scores['academic_score'] >= 40:
            reasoning.append(f"Strong academic performance ({scores['academic_score']:.1f}/50)")
        elif scores['academic_score'] < 25:
            reasoning.append(f"Weak academic performance ({scores['academic_score']:.1f}/50)")
        
        if scores['aptitude_score'] >= 40:
            reasoning.append(f"High aptitude match ({scores['aptitude_score']:.1f}/50)")
        
        if scores['personality_score'] >= 80:
            reasoning.append(f"Excellent personality fit ({scores['personality_score']:.1f}/100)")
        
        if scores['interest_score'] >= 30:
            reasoning.append(f"Strong interest alignment ({scores['interest_score']:.1f}/50)")
        
        return reasoning
    
    def _generate_suggestions(self, profile: StudentProfile, 
                             recommendation: StreamRecommendation) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if recommendation.match_percentage < 65:
            suggestions.append(f"Consider improving weak areas to increase {recommendation.stream.value} suitability")
        
        if recommendation.academic_score < 30:
            suggestions.append("Focus on improving academic performance in relevant subjects")
        
        if recommendation.aptitude_score < 30:
            suggestions.append("Consider aptitude test preparation courses")
        
        return suggestions
