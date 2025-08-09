import pandas as pd
import os
import logging
import yaml
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config_schema = {
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "properties": {
                "data_dir": {"type": "string"},
                "profiles_file": {"type": "string"},
                "liked_file": {"type": "string"},
                "matched_file": {"type": "string"},
                "blocked_file": {"type": "string"},
                "declined_file": {"type": "string"},
                "deleted_file": {"type": "string"},
                "reported_file": {"type": "string"},
                "required_columns": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["data_dir", "required_columns"]
        },
        "model": {
            "type": "object",
            "properties": {
                "models_dir": {"type": "string"},
                "max_tfidf_features": {"type": "integer"}
            },
            "required": ["models_dir"]
        },
        "preprocessing": {
            "type": "object",
            "properties": {
                "categorical_columns": {"type": "array", "items": {"type": "string"}},
                "tfidf_params": {
                    "type": "object",
                    "properties": {
                        "max_features": {"type": "integer"},
                        "stop_words": {"type": ["string", "null"]},
                        "min_df": {"type": "integer"}
                    },
                    "required": ["max_features"]
                },
                "keywords": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["categorical_columns", "tfidf_params", "keywords"]
        }
    },
    "required": ["data", "model", "preprocessing"]
}

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    default_config = {
        'data': {
            'data_dir': 'data',
            'profiles_file': 'Profiles.csv',
            'liked_file': 'LikedUsers.csv',
            'matched_file': 'MatchedUsers.csv',
            'blocked_file': 'BlockedUsers.csv',
            'declined_file': 'DeclinedUsers.csv',
            'deleted_file': 'DeletedUsers.csv',
            'reported_file': 'ReportedUsers.csv',
            'required_columns': [
                '__id__', 'userId', 'userName', 'age', 'country', 'language',
                'aboutMe', 'sex', 'seeking', 'relationshipGoals', 'subscribed',
                'subscribedEliteOne', 'subscribedEliteThree', 'subscribedEliteSix',
                'subscribedEliteTwelve'
            ]
        },
        'model': {
            'models_dir': 'models',
            'max_tfidf_features': 50
        },
        'preprocessing': {
            'categorical_columns': [
                'country', 'language', 'sex', 'seeking', 'relationshipGoals'
            ],
            'tfidf_params': {
                'max_features': 50,
                'stop_words': 'english',
                'min_df': 1
            },
            'keywords': [
                'love', 'soul mate', 'relationship', 'partner', 'soccer', 'football'
            ]
        }
    }
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validate(instance=config, schema=config_schema)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using default configuration")
        return default_config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return default_config
    except ValidationError as e:
        logger.error(f"Config validation error: {e}")
        return default_config

def validate_csv_schema(df, expected_cols, file_name):
    """Validate CSV file schema."""
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in {file_name}: {missing_cols}")
        raise ValueError(f"Missing columns in {file_name}: {missing_cols}")

def load_data(data_dir=None):
    """
    Load and merge CSV datasets.
    Returns: profiles_df, liked_df, matched_df, blocked_ids, declined_ids, deleted_ids, reported_ids
    """
    config = load_config()
    data_dir = data_dir or config['data']['data_dir']
    required_cols = config['data']['required_columns']
    start_time = config.get('start_time', None)  # Set by caller if needed

    try:
        profiles = pd.read_csv(
            os.path.join(data_dir, config['data']['profiles_file']),
            usecols=[c for c in required_cols if c in pd.read_csv(os.path.join(data_dir, config['data']['profiles_file']), nrows=1).columns]
        )
        validate_csv_schema(profiles, required_cols, config['data']['profiles_file'])
        
        liked = pd.read_csv(os.path.join(data_dir, config['data']['liked_file']), usecols=['userId', '__id__'])
        validate_csv_schema(liked, ['userId', '__id__'], config['data']['liked_file'])
        
        matched = pd.read_csv(os.path.join(data_dir, config['data']['matched_file']), usecols=['userId', '__id__'])
        validate_csv_schema(matched, ['userId', '__id__'], config['data']['matched_file'])
        
        blocked = pd.read_csv(os.path.join(data_dir, config['data']['blocked_file']), usecols=['__id__'])
        validate_csv_schema(blocked, ['__id__'], config['data']['blocked_file'])
        
        declined = pd.read_csv(os.path.join(data_dir, config['data']['declined_file']), usecols=['__id__'])
        validate_csv_schema(declined, ['__id__'], config['data']['declined_file'])
        
        deleted = pd.read_csv(os.path.join(data_dir, config['data']['deleted_file']), usecols=['__id__'])
        validate_csv_schema(deleted, ['__id__'], config['data']['deleted_file'])
        
        reported = pd.read_csv(os.path.join(data_dir, config['data']['reported_file']), usecols=['__id__'])
        validate_csv_schema(reported, ['__id__'], config['data']['reported_file'])
        
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        return None, None, None, None, None, None, None
    except ValueError as e:
        logger.error(f"Schema validation error: {e}")
        return None, None, None, None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return None, None, None, None, None, None, None

    profiles = profiles.drop_duplicates().fillna("unknown")
    logger.info(f"Loaded {len(profiles)} profiles, {len(liked)} liked, {len(matched)} matched")
    if start_time:
        logger.info(f"Data Loading Time: {time.time() - start_time:.2f} seconds")
    return (profiles, liked, matched, 
            blocked['__id__'].tolist(), declined['__id__'].tolist(), 
            deleted['__id__'].tolist(), reported['__id__'].tolist())