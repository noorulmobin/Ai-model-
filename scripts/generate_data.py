"""
Generate synthetic training data for FSC recommendation system
"""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_models import (
    StudentProfile, AcademicPerformance, AptitudeScores,
    InterestScores, PersonalityType
)


def generate_synthetic_profile(seed: int = None) -> StudentProfile:
    """Generate a synthetic student profile"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate academic performance with correlations
    base_ability = np.random.normal(70, 15)  # Base academic ability
    
    # Mathematics (correlated with base ability)
    math = np.clip(base_ability + np.random.normal(0, 8), 0, 100)
    
    # Science subjects (correlated with each other)
    science_base = base_ability + np.random.normal(0, 10)
    physics = np.clip(science_base + np.random.normal(0, 7), 0, 100)
    chemistry = np.clip(science_base + np.random.normal(0, 7), 0, 100)
    biology = np.clip(science_base + np.random.normal(0, 7), 0, 100)
    
    # Computer (correlated with math and logic)
    computer = np.clip((math + base_ability) / 2 + np.random.normal(0, 8), 0, 100)
    
    # Aggregate
    aggregate = np.mean([math, physics, chemistry, biology, computer])
    
    academic = AcademicPerformance(
        mathematics=math,
        biology=biology,
        physics=physics,
        chemistry=chemistry,
        computer=computer,
        aggregate=aggregate
    )
    
    # Generate aptitude scores (correlated with academic performance)
    math_apt = np.clip(math + np.random.normal(0, 10), 0, 100)
    science_apt = np.clip(np.mean([physics, chemistry, biology]) + np.random.normal(0, 10), 0, 100)
    verbal_apt = np.clip(np.random.normal(70, 12), 0, 100)
    logical_apt = np.clip(math_apt + np.random.normal(0, 8), 0, 100)
    spatial_apt = np.clip(np.random.normal(65, 12), 0, 100)
    
    aptitude = AptitudeScores(
        mathematical_ability=math_apt,
        scientific_aptitude=science_apt,
        verbal_ability=verbal_apt,
        logical_reasoning=logical_apt,
        spatial_ability=spatial_apt
    )
    
    # Generate interests (some correlation with abilities)
    medical_interest = np.clip(biology * 0.6 + np.random.normal(50, 20), 0, 100)
    engineering_interest = np.clip(math * 0.5 + physics * 0.3 + np.random.normal(40, 20), 0, 100)
    computer_interest = np.clip(computer * 0.7 + logical_apt * 0.2 + np.random.normal(30, 20), 0, 100)
    research_interest = np.clip(science_apt * 0.6 + np.random.normal(50, 20), 0, 100)
    creative_interest = np.clip(np.random.normal(50, 20), 0, 100)
    
    interests = InterestScores(
        medicine_healthcare=medical_interest,
        engineering_technology=engineering_interest,
        computers_programming=computer_interest,
        research_science=research_interest,
        creative_arts=creative_interest
    )
    
    # Random personality type
    personality = np.random.choice(list(PersonalityType))
    
    return StudentProfile(
        academic=academic,
        aptitude=aptitude,
        personality=personality,
        interests=interests,
        student_id=f"STU_{seed}" if seed is not None else None
    )


def generate_dataset(n_samples: int, output_path: str = None) -> pd.DataFrame:
    """Generate synthetic dataset"""
    print(f"Generating {n_samples} synthetic student profiles...")
    
    profiles = []
    for i in range(n_samples):
        profile = generate_synthetic_profile(seed=i)
        profiles.append(profile)
    
    # Convert to DataFrame
    data = []
    for profile in profiles:
        row = {
            'student_id': profile.student_id,
            'math': profile.academic.mathematics,
            'bio': profile.academic.biology,
            'physics': profile.academic.physics,
            'chem': profile.academic.chemistry,
            'computer': profile.academic.computer,
            'aggregate': profile.academic.aggregate,
            'apt_math': profile.aptitude.mathematical_ability,
            'apt_science': profile.aptitude.scientific_aptitude,
            'apt_verbal': profile.aptitude.verbal_ability,
            'apt_logical': profile.aptitude.logical_reasoning,
            'apt_spatial': profile.aptitude.spatial_ability,
            'int_medical': profile.interests.medicine_healthcare,
            'int_engineering': profile.interests.engineering_technology,
            'int_computer': profile.interests.computers_programming,
            'int_research': profile.interests.research_science,
            'int_arts': profile.interests.creative_arts,
            'personality': profile.personality.value
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
    
    return df, profiles


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/synthetic/training_data.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    df, profiles = generate_dataset(args.samples, args.output)
    
    print(f"\nDataset Statistics:")
    print(df.describe())
    print(f"\nPersonality Distribution:")
    print(df['personality'].value_counts())


if __name__ == '__main__':
    main()
