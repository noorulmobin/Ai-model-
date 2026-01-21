"""
ML model training for FSC recommendation system
Supports multiple model approaches
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path

from src.data_models import StudentProfile, StreamType
from src.feature_engineer import FeatureEngineer
from src.scoring import RuleBasedScorer


class ModelTrainer:
    """Train and evaluate ML models for stream recommendation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.scorer = RuleBasedScorer()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, profiles: List[StudentProfile], 
                    use_rule_based_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data
        
        Args:
            profiles: List of student profiles
            use_rule_based_labels: If True, use rule-based scorer for labels/scores
        
        Returns:
            X: Feature matrix
            y_classification: Class labels (for classification)
            y_regression: Scores for each stream (for regression)
        """
        # Extract features
        X = self.feature_engineer.extract_features_batch(profiles)
        X = self.scaler.fit_transform(X)
        
        if use_rule_based_labels:
            # Use rule-based scorer to generate labels
            recommendations = []
            for profile in profiles:
                result = self.scorer.recommend(profile)
                recommendations.append(result.top_recommendation.stream)
            
            y_classification = self.feature_engineer.extract_target_labels(
                profiles, recommendations
            )
            
            # Generate scores for each stream
            y_regression = self.feature_engineer.extract_target_scores(profiles, self.scorer)
        else:
            # If you have ground truth labels, use them here
            raise NotImplementedError("Ground truth labels not implemented yet")
        
        return X, y_classification, y_regression
    
    def train_classification_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train multiple classification models"""
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state, multi_class='multinomial'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf', probability=True, random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state, eval_metric='mlogloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state, verbose=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'predictions': val_pred
            }
            
            print(f"{name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        return results
    
    def train_regression_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train regression models to predict scores for each stream"""
        models = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'svr': MultiOutputRegressor(SVR(kernel='rbf')),  # SVR needs wrapper for multi-output
            'xgboost': xgb.XGBRegressor(random_state=self.random_state),
            'lightgbm': MultiOutputRegressor(lgb.LGBMRegressor(random_state=self.random_state, verbose=-1))  # LightGBM needs wrapper for multi-output
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'predictions': val_pred
            }
            
            print(f"{name} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")
        
        return results
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      base_models: Dict) -> Dict:
        """Train ensemble model combining multiple base models"""
        # Create voting classifier from best models
        estimators = [(name, model['model']) for name, model in base_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        print("Training ensemble model...")
        ensemble.fit(X_train, y_train)
        
        val_pred = ensemble.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"Ensemble - Val Acc: {val_acc:.4f}")
        
        return {
            'model': ensemble,
            'val_accuracy': val_acc,
            'predictions': val_pred
        }
    
    def train_all_models(self, profiles: List[StudentProfile], 
                        test_size: float = 0.2) -> Dict:
        """Train all model types and return best model"""
        print("Preparing data...")
        X, y_class, y_reg = self.prepare_data(profiles)
        
        # Split data - use same random_state to ensure consistent splits
        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg, test_size=test_size, random_state=self.random_state, stratify=y_class
        )
        
        # Split train into train/val for classification
        X_train, X_val, y_train_class, y_val_class, y_train_reg, y_val_reg = train_test_split(
            X_train, y_train_class, y_train_reg, test_size=test_size, random_state=self.random_state, stratify=y_train_class
        )
        
        print(f"Training set: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Train classification models
        print("\n=== Training Classification Models ===")
        class_results = self.train_classification_models(
            X_train, y_train_class, X_val, y_val_class
        )
        
        # Train regression models
        print("\n=== Training Regression Models ===")
        reg_results = self.train_regression_models(
            X_train, y_train_reg, X_val, y_val_reg
        )
        
        # Find best classification model
        best_class_name = max(class_results.keys(), 
                            key=lambda k: class_results[k]['val_accuracy'])
        best_class_model = class_results[best_class_name]['model']
        
        print(f"\nBest Classification Model: {best_class_name} "
              f"(Val Acc: {class_results[best_class_name]['val_accuracy']:.4f})")
        
        # Evaluate on test set
        test_pred = best_class_model.predict(X_test)
        test_acc = accuracy_score(y_test_class, test_pred)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_class, test_pred, 
                                  target_names=[s.value for s in StreamType]))
        
        # Store results
        self.models = {
            'classification': class_results,
            'regression': reg_results,
            'best_classification': best_class_name
        }
        self.best_model = best_class_model
        self.best_model_name = best_class_name
        
        return {
            'models': self.models,
            'best_model': best_class_model,
            'best_model_name': best_class_name,
            'test_accuracy': test_acc,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer
        }
    
    def save_model(self, model_path: str, scaler_path: str):
        """Save trained model and scaler"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str):
        """Load trained model and scaler"""
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
