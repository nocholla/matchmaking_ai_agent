import joblib
import os
import pandas as pd

def save_models(model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx, models_dir="models"):
    """
    Save trained models and encoders to models_dir.
    """
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "matchmaking_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(user_to_idx, os.path.join(models_dir, "user_to_idx.pkl"))
    joblib.dump(profile_to_idx, os.path.join(models_dir, "profile_to_idx.pkl"))

def save_recommendations(top_matches, output_dir="data"):
    """
    Save top matches to CSV.
    """
    recommendations = top_matches[['__id__', 'userName', 'final_score']].copy()
    recommendations['reasons'] = top_matches.apply(
        lambda row: [
            "Country match" if row['country_match'] else None,
            "Language match" if row['language_match'] else None,
            "Relationship goals match" if row['goal_match'] else None,
            "High ML compatibility" if row['ml_score'] > 0.5 else None,
            "Subscribed user" if row['subscribed_score'] > 0 else None
        ], axis=1)
    os.makedirs(output_dir, exist_ok=True)
    recommendations.to_csv(os.path.join(output_dir, "recommendations.csv"), index=False)
