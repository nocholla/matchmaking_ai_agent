import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import joblib
from src.recommender import train_model, predict_compatibility, load_model_and_encoders

@pytest.fixture
def mock_data():
    """Create mock interaction matrix and features."""
    interaction_matrix = csr_matrix(([1, 2], ([0, 1], [0, 1])), shape=(2, 2))
    X_features = csr_matrix([[1, 2, 3], [4, 5, 6]])
    return interaction_matrix, X_features

@pytest.fixture
def mock_model_files(tmp_path):
    """Create mock model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    mock_model = {'dummy': 'model'}
    mock_scaler = {'dummy': 'scaler'}
    mock_encoders = {'country': 'encoder'}
    mock_tfidf = {'dummy': 'tfidf'}
    mock_user_to_idx = {'user1': 0}
    mock_profile_to_idx = {'profile1': 0}
    
    joblib.dump(mock_model, model_dir / "matchmaking_model.pkl")
    joblib.dump(mock_scaler, model_dir / "scaler.pkl")
    joblib.dump(mock_encoders, model_dir / "label_encoders.pkl")
    joblib.dump(mock_tfidf, model_dir / "tfidf_vectorizer.pkl")
    joblib.dump(mock_user_to_idx, model_dir / "user_to_idx.pkl")
    joblib.dump(mock_profile_to_idx, model_dir / "profile_to_idx.pkl")
    
    return str(model_dir)

def test_train_model(mock_data):
    """Test train_model function."""
    interaction_matrix, X_features = mock_data
    model, scaler = train_model(interaction_matrix, X_features)
    assert model is not None, "Model should be trained"
    assert scaler is not None, "Scaler should be fitted"

def test_predict_compatibility(mock_data):
    """Test predict_compatibility function."""
    interaction_matrix, X_features = mock_data
    model, scaler = train_model(interaction_matrix, X_features)
    profiles = pd.DataFrame({
        '__id__': [0, 1],
        'userName': ['Alice', 'Bob'],
        'ml_score': [0.0, 0.0],
        'country_match': [True, False],
        'language_match': [True, False],
        'goal_match': [True, False]
    })
    user_to_idx = {'user123': 0}
    profile_to_idx = {0: 0, 1: 1}
    result = predict_compatibility(model, scaler, 'user123', profiles, X_features, user_to_idx, profile_to_idx)
    assert 'ml_score' in result.columns, "Result should have ml_score"
    assert 'final_score' in result.columns, "Result should have final_score"

def test_load_model_and_encoders(mock_model_files):
    """Test load_model_and_encoders function."""
    model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx = load_model_and_encoders(models_dir=mock_model_files)
    assert model is not None, "Model should load"
    assert scaler is not None, "Scaler should load"
    assert label_encoders is not None, "Label encoders should load"
    assert tfidf is not None, "TF-IDF vectorizer should load"
    assert user_to_idx is not None, "User-to-idx should load"
    assert profile_to_idx is not None, "Profile-to-idx should load"

def test_load_model_and_encoders_missing():
    """Test load_model_and_encoders with missing files."""
    model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx = load_model_and_encoders(models_dir="nonexistent")
    assert model is None, "Model should be None for missing files"
    assert scaler is None, "Scaler should be None for missing files"
    assert label_encoders is None, "Label encoders should be None for missing files"
    assert tfidf is None, "TF-IDF vectorizer should be None for missing files"
    assert user_to_idx is None, "User-to-idx should be None for missing files"
    assert profile_to_idx is None, "Profile-to-idx should be None for missing files"