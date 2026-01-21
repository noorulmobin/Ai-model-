"""
Example usage of FSC Recommendation System
"""
from src.data_models import (
    StudentProfile, AcademicPerformance, AptitudeScores,
    InterestScores, PersonalityType
)
from src.predictor import FSCPredictor
from src.scoring import RuleBasedScorer


def example_usage():
    """Example of using the recommendation system"""
    
    # Create a sample student profile
    profile = StudentProfile(
        academic=AcademicPerformance(
            mathematics=82,
            biology=76,
            physics=71,
            chemistry=74,
            computer=79,
            aggregate=76.4
        ),
        aptitude=AptitudeScores(
            mathematical_ability=85,
            scientific_aptitude=78,
            verbal_ability=72,
            logical_reasoning=81,
            spatial_ability=68
        ),
        personality=PersonalityType.INTP,
        interests=InterestScores(
            medicine_healthcare=30,
            engineering_technology=75,
            computers_programming=90,
            research_science=60,
            creative_arts=40
        ),
        student_id="STU_001"
    )
    
    # Method 1: Use rule-based system (no training required)
    print("="*60)
    print("Rule-Based Recommendation System")
    print("="*60)
    
    scorer = RuleBasedScorer()
    result = scorer.recommend(profile)
    
    print(f"\nTop Recommendation: {result.top_recommendation.stream.value}")
    print(f"Match Percentage: {result.top_recommendation.match_percentage:.2f}%")
    print(f"Suitability: {result.top_recommendation.suitability_class}")
    
    print("\nAll Recommendations:")
    for rec in result.recommendations:
        print(f"  {rec.stream.value}: {rec.match_percentage:.2f}% ({rec.suitability_class})")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Method 2: Use ML model (requires training first)
    print("\n" + "="*60)
    print("ML Model Recommendation System")
    print("="*60)
    
    try:
        predictor = FSCPredictor(
            model_path='models/trained/best_model.pkl',
            scaler_path='models/trained/scaler.pkl'
        )
        
        recommendation = predictor.recommend(profile, use_ml=True)
        
        print(f"\nML Prediction: {recommendation['ml_prediction']}")
        print("\nML Probabilities:")
        for stream, prob in recommendation['ml_probabilities'].items():
            print(f"  {stream}: {prob:.2%}")
        
        print(f"\nRule-Based Recommendation: {recommendation['rule_based_recommendation']}")
        
    except FileNotFoundError:
        print("\nML model not found. Please train the model first:")
        print("  python train.py --generate --samples 10000")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    example_usage()
