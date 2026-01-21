"""
Evaluation script for FSC Recommendation Model
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from scripts.generate_data import generate_dataset
from src.model_trainer import ModelTrainer
from src.predictor import FSCPredictor
from src.data_models import StreamType


def load_profiles_from_dataframe(df):
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


def evaluate_model(model_path: str, scaler_path: str, test_data_path: str):
    """Evaluate trained model on test data"""
    print("Loading model...")
    predictor = FSCPredictor(model_path, scaler_path)
    
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    profiles = load_profiles_from_dataframe(df)
    
    print(f"Evaluating on {len(profiles)} test samples...")
    
    # Get ground truth (rule-based recommendations)
    from src.scoring import RuleBasedScorer
    scorer = RuleBasedScorer()
    
    y_true = []
    y_pred = []
    
    for profile in profiles:
        # Ground truth
        result = scorer.recommend(profile)
        true_stream = result.top_recommendation.stream
        
        # Prediction
        pred_stream = predictor.predict_stream(profile)
        
        y_true.append(true_stream.value)
        y_pred.append(pred_stream.value)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=[s.value for s in StreamType]))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, 
                         labels=[s.value for s in StreamType])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[s.value for s in StreamType],
                yticklabels=[s.value for s in StreamType])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation_confusion_matrix.png')
    print("\nConfusion matrix saved to evaluation_confusion_matrix.png")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate FSC Recommendation Model')
    parser.add_argument('--model', type=str, default='models/trained/best_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--scaler', type=str, default='models/trained/scaler.pkl',
                       help='Path to scaler')
    parser.add_argument('--test-data', type=str, default='data/synthetic/test_data.csv',
                       help='Path to test data')
    parser.add_argument('--generate-test', action='store_true',
                       help='Generate test data if not exists')
    parser.add_argument('--test-samples', type=int, default=2000,
                       help='Number of test samples to generate')
    
    args = parser.parse_args()
    
    # Generate test data if needed
    if args.generate_test or not Path(args.test_data).exists():
        print("Generating test data...")
        df, _ = generate_dataset(args.test_samples, args.test_data)
    
    # Evaluate
    evaluate_model(args.model, args.scaler, args.test_data)


if __name__ == '__main__':
    main()
