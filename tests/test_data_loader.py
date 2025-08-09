import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import os
import yaml
from src.data_loader import load_data, load_config, validate_csv_schema

@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory with mock CSV files."""
    data_path = tmp_path / "data"
    data_path.mkdir()
    
    profiles_data = pd.DataFrame({
        '__id__': [1, 2],
        'userId': ['user1', 'user2'],
        'userName': ['Alice', 'Bob'],
        'age': [25, 30],
        'country': ['Kenya', 'Nigeria'],
        'language': ['Swahili', 'English'],
        'aboutMe': ['Love soccer', 'Enjoy music'],
        'sex': ['Female', 'Male'],
        'seeking': ['Male', 'Female'],
        'relationshipGoals': ['Long-term', 'Casual'],
        'subscribed': [True, False],
        'subscribedEliteOne': [False, False],
        'subscribedEliteThree': [False, False],
        'subscribedEliteSix': [False, False],
        'subscribedEliteTwelve': [False, False]
    })
    profiles_data.to_csv(data_path / "Profiles.csv", index=False)
    
    for file in ['LikedUsers.csv', 'MatchedUsers.csv', 'BlockedUsers.csv', 
                 'DeclinedUsers.csv', 'DeletedUsers.csv', 'ReportedUsers.csv']:
        pd.DataFrame({'__id__': [], 'userId': []}).to_csv(data_path / file, index=False)
    
    return str(data_path)

@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config.yaml file."""
    config_path = tmp_path / "config.yaml"
    config = {
        "data": {
            "data_dir": "data",
            "profiles_file": "Profiles.csv",
            "liked_file": "LikedUsers.csv",
            "matched_file": "MatchedUsers.csv",
            "blocked_file": "BlockedUsers.csv",
            "declined_file": "DeclinedUsers.csv",
            "deleted_file": "DeletedUsers.csv",
            "reported_file": "ReportedUsers.csv",
            "required_columns": [
                "__id__", "userId", "userName", "age", "country", "language",
                "aboutMe", "sex", "seeking", "relationshipGoals", "subscribed",
                "subscribedEliteOne", "subscribedEliteThree", "subscribedEliteSix",
                "subscribedEliteTwelve"
            ]
        },
        "model": {
            "models_dir": "models",
            "max_tfidf_features": 50
        },
        "preprocessing": {
            "categorical_columns": [
                "country", "language", "sex", "seeking", "relationshipGoals"
            ],
            "tfidf_params": {
                "max_features": 50,
                "stop_words": "english",
                "min_df": 1
            },
            "keywords": [
                "love", "soul mate", "relationship", "partner", "soccer", "football"
            ]
        }
    }
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    return str(config_path)

def test_load_data(data_dir, config_file):
    """Test the load_data function."""
    result = load_data(data_dir=data_dir)
    profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids = result
    
    assert profiles is not None, "Profiles DataFrame should not be None"
    assert isinstance(profiles, pd.DataFrame), "Profiles should be a DataFrame"
    assert len(profiles) == 2, "Expected 2 profiles"
    assert set(profiles.columns).issuperset(['__id__', 'userId', 'userName', 'age']), "Missing required columns"
    assert liked.empty, "LikedUsers should be empty"
    assert matched.empty, "MatchedUsers should be empty"
    assert blocked_ids == [], "Blocked IDs should be empty"

def test_load_config(config_file):
    """Test the load_config function."""
    config = load_config(config_path=config_file)
    assert config is not None, "Config should not be None"
    assert isinstance(config, dict), "Config should be a dictionary"
    assert "data" in config, "Config should have 'data' key"
    assert config["data"]["data_dir"] == "data", "Data dir should be 'data'"
    assert "model" in config, "Config should have 'model' key"
    assert config["model"]["models_dir"] == "models", "Models dir should be 'models'"
    assert "required_columns" in config["data"], "Config should have 'required_columns'"
    assert len(config["data"]["required_columns"]) == 15, "Expected 15 required columns"
    assert "preprocessing" in config, "Config should have 'preprocessing' key"
    assert config["preprocessing"]["categorical_columns"] == ["country", "language", "sex", "seeking", "relationshipGoals"], "Incorrect categorical_columns"
    assert config["preprocessing"]["tfidf_params"]["max_features"] == 50, "Incorrect tfidf max_features"
    assert "keywords" in config["preprocessing"], "Config should have 'keywords'"

def test_load_config_missing():
    """Test load_config with missing file."""
    config = load_config(config_path="nonexistent.yaml")
    assert config is not None, "Config should not be None for missing file"
    assert isinstance(config, dict), "Config should be a dictionary"
    assert "preprocessing" in config, "Default config should have 'preprocessing'"
    assert "categorical_columns" in config["preprocessing"], "Default config should have 'categorical_columns'"