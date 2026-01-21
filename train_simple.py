"""
Simple training script with multiple options
Run this to see different training approaches
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.model_trainer import ModelTrainer
from scripts.generate_data import generate_dataset
from train import load_profiles_from_dataframe
import pandas as pd


def train_quick_test():
    """Quick test with small dataset (2-3 minutes)"""
    print("="*60)
    print("QUICK TEST TRAINING (1000 samples)")
    print("="*60)
    
    # Generate small dataset
    print("\n1. Generating 1000 synthetic profiles...")
    df, _ = generate_dataset(1000, 'data/synthetic/quick_test.csv')
    
    # Load profiles
    print("2. Loading profiles...")
    profiles = load_profiles_from_dataframe(df)
    
    # Train
    print("3. Training models...")
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_all_models(profiles, test_size=0.2)
    
    print(f"\n✅ Quick test complete!")
    print(f"   Best model: {results['best_model_name']}")
    print(f"   Accuracy: {results['test_accuracy']:.4f}")
    
    return results


def train_standard():
    """Standard training with 10,000 samples (15-30 minutes)"""
    print("="*60)
    print("STANDARD TRAINING (10,000 samples)")
    print("="*60)
    
    # Generate dataset
    print("\n1. Generating 10,000 synthetic profiles...")
    df, _ = generate_dataset(10000, 'data/synthetic/training_data.csv')
    
    # Load profiles
    print("2. Loading profiles...")
    profiles = load_profiles_from_dataframe(df)
    
    # Train
    print("3. Training all models...")
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_all_models(profiles, test_size=0.2)
    
    # Save
    print("4. Saving models...")
    trainer.save_model('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
    
    print(f"\n✅ Standard training complete!")
    print(f"   Best model: {results['best_model_name']}")
    print(f"   Accuracy: {results['test_accuracy']:.4f}")
    print(f"   Models saved to models/trained/")
    
    return results


def train_large():
    """Large dataset training with 20,000 samples (30-60 minutes)"""
    print("="*60)
    print("LARGE DATASET TRAINING (20,000 samples)")
    print("="*60)
    
    # Generate dataset
    print("\n1. Generating 20,000 synthetic profiles...")
    df, _ = generate_dataset(20000, 'data/synthetic/large_training_data.csv')
    
    # Load profiles
    print("2. Loading profiles...")
    profiles = load_profiles_from_dataframe(df)
    
    # Train
    print("3. Training all models (this may take a while)...")
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_all_models(profiles, test_size=0.2)
    
    # Save
    print("4. Saving models...")
    trainer.save_model('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
    
    print(f"\n✅ Large dataset training complete!")
    print(f"   Best model: {results['best_model_name']}")
    print(f"   Accuracy: {results['test_accuracy']:.4f}")
    
    return results


def train_with_existing_data(data_path):
    """Train with your own CSV data"""
    print("="*60)
    print(f"TRAINING WITH YOUR DATA: {data_path}")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Found {len(df)} records")
    
    # Load profiles
    print("2. Converting to student profiles...")
    profiles = load_profiles_from_dataframe(df)
    
    # Train
    print("3. Training models...")
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_all_models(profiles, test_size=0.2)
    
    # Save
    print("4. Saving models...")
    trainer.save_model('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {results['best_model_name']}")
    print(f"   Accuracy: {results['test_accuracy']:.4f}")
    
    return results


def compare_models():
    """Compare all trained models side by side"""
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Generate data
    print("\n1. Generating test data...")
    df, _ = generate_dataset(5000, 'data/synthetic/comparison_data.csv')
    profiles = load_profiles_from_dataframe(df)
    
    # Train
    print("2. Training all models...")
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_all_models(profiles, test_size=0.2)
    
    # Compare
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Status':<10}")
    print("-"*60)
    
    for name, metrics in results['models']['classification'].items():
        status = "✅ Best" if name == results['best_model_name'] else ""
        print(f"{name:<20} {metrics['train_accuracy']:<12.4f} {metrics['val_accuracy']:<12.4f} {status:<10}")
    
    print(f"\n🏆 Best Model: {results['best_model_name']}")
    print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
    
    return results


def main():
    """Main menu for training options"""
    print("\n" + "="*60)
    print("FSC RECOMMENDATION SYSTEM - TRAINING OPTIONS")
    print("="*60)
    print("\nChoose a training option:")
    print("1. Quick Test (1000 samples, ~3 minutes)")
    print("2. Standard Training (10,000 samples, ~20 minutes)")
    print("3. Large Dataset (20,000 samples, ~45 minutes)")
    print("4. Train with Your Own Data")
    print("5. Compare All Models")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        train_quick_test()
    elif choice == '2':
        train_standard()
    elif choice == '3':
        train_large()
    elif choice == '4':
        data_path = input("Enter path to your CSV file: ").strip()
        if Path(data_path).exists():
            train_with_existing_data(data_path)
        else:
            print(f"❌ File not found: {data_path}")
    elif choice == '5':
        compare_models()
    elif choice == '6':
        print("Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please run again and select 1-6.")
        return
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Test the model: python example_usage.py")
    print("2. Evaluate: python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl")
    print("3. Use in your code: from src.predictor import FSCPredictor")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
