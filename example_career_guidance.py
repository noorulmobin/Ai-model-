"""
Example: Using Career Guidance Q&A with FSC Recommendations
"""
from src.data_models import (
    StudentProfile, AcademicPerformance, AptitudeScores,
    InterestScores, PersonalityType
)
from src.predictor import FSCPredictor
from src.career_guidance import CareerGuidanceQA


def example_with_career_guidance():
    """Example showing career guidance integration"""
    
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
    
    print("="*60)
    print("FSC Recommendation with Career Guidance")
    print("="*60)
    
    # Initialize predictor with career guidance
    try:
        predictor = FSCPredictor(
            model_path='models/trained/best_model.pkl',
            scaler_path='models/trained/scaler.pkl',
            use_career_guidance=True
        )
        use_ml = True
    except FileNotFoundError:
        print("ML model not found, using rule-based system...")
        predictor = FSCPredictor(use_career_guidance=True)
        use_ml = False
    
    # Get recommendation with career guidance
    result = predictor.recommend(profile, use_ml=use_ml)
    
    # Display recommendation
    if use_ml:
        print(f"\n🎯 ML Prediction: {result['ml_prediction']}")
        print(f"📊 Rule-Based: {result['rule_based_recommendation']}")
    else:
        print(f"\n🎯 Recommended Stream: {result['recommended_stream']}")
        print(f"📊 Match Percentage: {result['match_percentage']:.1f}%")
    
    # Display career guidance if available
    if 'career_guidance' in result and result['career_guidance']:
        print("\n" + "="*60)
        print("💼 Career Guidance Q&A")
        print("="*60)
        
        for i, qa in enumerate(result['career_guidance'], 1):
            print(f"\n{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
    
    # Display general career advice
    if 'general_career_advice' in result and result['general_career_advice']:
        print("\n" + "="*60)
        print("📚 General Career Advice")
        print("="*60)
        
        for i, qa in enumerate(result['general_career_advice'][:3], 1):
            print(f"\n{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer'][:200]}...")  # Truncate long answers
    
    print("\n" + "="*60)


def example_direct_qa_search():
    """Example of directly using career guidance Q&A"""
    
    print("="*60)
    print("Direct Career Guidance Q&A Search")
    print("="*60)
    
    # Initialize career guidance
    career_qa = CareerGuidanceQA()
    
    # Search for specific topics
    queries = [
        "What career options are available after Pre-Engineering?",
        "How to choose between Pre-Medical and Pre-Engineering?",
        "What skills are needed for computer science?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = career_qa.search_qa(query, top_k=2)
        
        if results:
            for i, qa in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Q: {qa['question']}")
                print(f"    A: {qa['answer'][:150]}...")
        else:
            print("  No results found")
    
    # Get stream-specific guidance
    print("\n" + "="*60)
    print("Stream-Specific Guidance")
    print("="*60)
    
    for stream in ["Pre-Engineering", "Pre-Medical", "ICS"]:
        print(f"\n📘 {stream}:")
        guidance = career_qa.get_stream_guidance(stream)
        
        if guidance:
            for i, qa in enumerate(guidance[:2], 1):
                print(f"  {i}. {qa['question']}")
        else:
            print("  No specific guidance available")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    print("\nExample 1: Recommendation with Career Guidance")
    print("-" * 60)
    example_with_career_guidance()
    
    print("\n\nExample 2: Direct Q&A Search")
    print("-" * 60)
    example_direct_qa_search()
