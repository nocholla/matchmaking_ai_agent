import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_user_profile(user_profile):
    """Validate user profile inputs."""
    required_keys = ['userId', 'age', 'sex', 'seeking', 'country', 'language', 'relationshipGoals', 'aboutMe']
    for key in required_keys:
        if key not in user_profile:
            logger.error(f"Missing required user profile key: {key}")
            raise ValueError(f"Missing required user profile key: {key}")
    if not isinstance(user_profile['age'], (int, float)) or user_profile['age'] < 18 or user_profile['age'] > 70:
        logger.error(f"Invalid age: {user_profile['age']}")
        raise ValueError("Age must be a number between 18 and 70")
    if user_profile['sex'] not in ['Female', 'Male', 'unknown']:
        logger.error(f"Invalid sex: {user_profile['sex']}")
        raise ValueError("Sex must be 'Female', 'Male', or 'unknown'")
    if user_profile['seeking'] not in ['Female', 'Male', 'unknown']:
        logger.error(f"Invalid seeking: {user_profile['seeking']}")
        raise ValueError("Seeking must be 'Female', 'Male', or 'unknown'")

def apply_rules(profiles, user_profile, blocked_ids, declined_ids, deleted_ids, reported_ids):
    """
    Apply rule-based filtering to profiles.
    Returns: filtered profiles with match scores
    """
    try:
        validate_user_profile(user_profile)
        filtered = profiles.copy()
        
        # Exclude blocked, declined, deleted, reported users
        excluded_ids = set(blocked_ids).union(declined_ids, deleted_ids, reported_ids)
        filtered = filtered[~filtered['__id__'].isin(excluded_ids)]
        
        # Match seeking and sex
        if user_profile.get('seeking') and user_profile.get('sex'):
            filtered = filtered[filtered['sex'] == user_profile['seeking']]
            filtered = filtered[filtered['seeking'] == user_profile['sex']]
        
        # Age range (±5 years)
        if user_profile.get('age'):
            age = float(user_profile['age'])
            filtered = filtered[filtered['age'].between(age - 5, age + 5)]
        
        # Prefer same country, language, or relationship goals
        filtered['country_match'] = filtered['country'] == user_profile.get('country', 'unknown')
        filtered['language_match'] = filtered['language'] == user_profile.get('language', 'unknown')
        filtered['goal_match'] = filtered['relationshipGoals'] == user_profile.get('relationshipGoals', 'unknown')
        
        # Prioritize subscribed users
        filtered['subscribed_score'] = filtered[['subscribed', 'subscribedEliteOne', 
                                                'subscribedEliteThree', 'subscribedEliteSix', 
                                                'subscribedEliteTwelve']].sum(axis=1)
        
        logger.info(f"Filtered to {len(filtered)} profiles after applying rules")
        return filtered
    
    except Exception as e:
        logger.error(f"Error in apply_rules: {e}")
        raise

def encode_user_profile(user_profile, label_encoders, tfidf):
    """
    Encode user profile for ML prediction.
    Returns: encoded user features
    """
    try:
        validate_user_profile(user_profile)
        user_features = []
        for col in ['country', 'language', 'sex', 'seeking', 'relationshipGoals']:
            if col in label_encoders and user_profile[col] != 'unknown':
                try:
                    user_features.append(label_encoders[col].transform([user_profile[col]])[0])
                except:
                    user_features.append(label_encoders[col].transform(['unknown'])[0])
            else:
                user_features.append(0)
        
        user_features.extend([
            user_profile['age'],
            0, 0, 0, 0, 0,  # Subscribed flags
            sum(1 for word in ['love', 'soul mate', 'relationship', 'partner', 'soccer', 'football'] 
                if word.lower() in user_profile['aboutMe'].lower()) / 6.0
        ])
        
        tfidf_vec = tfidf.transform([user_profile['aboutMe']]).toarray().flatten()
        encoded_features = np.concatenate([user_features, tfidf_vec])
        logger.info(f"Encoded user profile with {len(encoded_features)} features")
        return encoded_features
    
    except Exception as e:
        logger.error(f"Error in encode_user_profile: {e}")
        raise