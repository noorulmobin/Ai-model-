"""
Main training script for FSC Recommendation System
"""
import argparse
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent))

from scripts.generate_data import generate_dataset
from src.model_trainer import ModelTrainer
from src.data_models import StudentProfile


def load_profiles_from_dataframe(df) -> list:
    """Load student profiles from DataFrame"""
    from src.data_models import (
        AcademicPerformance, AptitudeScores, InterestScores, PersonalityType
    )
    
    profiles = []
    for _, row in df.iterrows():
        academic = AcademicPerformance(
            mathematics=row['math'],
            biology=row['bio'],
            physics=row['physics'],
            chemistry=row['chem'],
            computer=row['computer'],
            aggregate=row['aggregate']
        )
        
        aptitude = AptitudeScores(
            mathematical_ability=row['apt_math'],
            scientific_aptitude=row['apt_science'],
            verbal_ability=row['apt_verbal'],
            logical_reasoning=row['apt_logical'],
            spatial_ability=row['apt_spatial']
        )
        
        interests = InterestScores(
            medicine_healthcare=row['int_medical'],
            engineering_technology=row['int_engineering'],
            computers_programming=row['int_computer'],
            research_science=row['int_research'],
            creative_arts=row['int_arts']
        )
        
        personality = PersonalityType(row['personality'])
        
        profile = StudentProfile(
            academic=academic,
            aptitude=aptitude,
            personality=personality,
            interests=interests,
            student_id=row.get('student_id')
        )
        
        profiles.append(profile)
    
    return profiles


def main():
    parser = argparse.ArgumentParser(description='Train FSC Recommendation Model')
    parser.add_argument('--data', type=str, default='data/synthetic/training_data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--generate', action='store_true',
                       help='Generate synthetic data if not exists')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--model-dir', type=str, default='models/trained',
                       help='Directory to save trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--use-career-guidance', action='store_true', default=True,
                       help='Use career guidance features in training (default: True)')
    parser.add_argument('--no-career-guidance', dest='use_career_guidance', action='store_false',
                       help='Disable career guidance features')
    
    args = parser.parse_args()
    
    # Generate data if needed
    if args.generate or not Path(args.data).exists():
        print("Generating synthetic training data...")
        df, _ = generate_dataset(args.samples, args.data)
    else:
        print(f"Loading data from {args.data}...")
        import pandas as pd
        df = pd.read_csv(args.data)
    
    # Load profiles
    print("Loading student profiles...")
    profiles = load_profiles_from_dataframe(df)
    print(f"Loaded {len(profiles)} student profiles")
    
    # Train models
    print("\n" + "="*50)
    print("Starting Model Training")
    print("="*50)
    
    if args.use_career_guidance:
        print("✅ Career guidance features: ENABLED")
        print("   Training with enhanced features (original + career guidance)")
    else:
        print("ℹ️  Career guidance features: DISABLED")
        print("   Training with original features only")
    
    trainer = ModelTrainer(
        random_state=42,
        use_career_guidance=args.use_career_guidance
    )
    results = trainer.train_all_models(profiles, test_size=args.test_size)
    
    # Save models
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'best_model.pkl'
    scaler_path = model_dir / 'scaler.pkl'
    metadata_path = model_dir / 'training_metadata.pkl'
    
    trainer.save_model(str(model_path), str(scaler_path))
    
    # Save metadata
    metadata = {
        'best_model_name': results['best_model_name'],
        'test_accuracy': results['test_accuracy'],
        'model_results': results['models'],
        'feature_names': trainer.feature_engineer.get_feature_names()
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nTraining completed!")
    print(f"Best Model: {results['best_model_name']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"\nModels saved to {model_dir}")


if __name__ == '__main__':
    main()
