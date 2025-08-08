import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.recommender import train_model, predict_compatibility
from src.agent import apply_rules, encode_user_profile
from src.utils import save_models, save_recommendations

def main():
    parser = argparse.ArgumentParser(description="Africa Love Match - Matchmaking AI Agent CLI")
    parser.add_argument("--user_id", default="user123", help="User ID")
    parser.add_argument("--age", type=int, default=25, help="Age (18-70)")
    parser.add_argument("--sex", default="unknown", choices=["Female", "Male", "unknown"], help="Sex")
    parser.add_argument("--seeking", default="unknown", choices=["Male", "Female", "unknown"], help="Seeking")
    parser.add_argument("--country", default="unknown", help="Country")
    parser.add_argument("--language", default="unknown", help="Language")
    parser.add_argument("--relationship_goals", default="unknown", help="Relationship Goals")
    parser.add_argument("--about_me", default="Looking for true love and enjoy soccer", help="About Me")

    args = parser.parse_args()

    # User profile
    user_profile = {
        'userId': args.user_id,
        'age': args.age,
        'sex': args.sex,
        'seeking': args.seeking,
        'country': args.country,
        'language': args.language,
        'relationshipGoals': args.relationship_goals,
        'aboutMe': args.about_me
    }

    # Load data
    (profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids) = load_data(data_dir="data")
    if profiles is None:
        print("Error: Failed to load data. Check CSV files in data/")
        return

    # Preprocess data
    (profiles, interaction_matrix, X_features, user_to_idx, profile_to_idx, 
     label_encoders, tfidf) = preprocess_data(profiles, liked, matched)

    # Train model
    model, scaler = train_model(interaction_matrix, X_features)

    # Save models
    save_models(model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx, models_dir="models")

    # Apply rules
    filtered_profiles = apply_rules(profiles, user_profile, blocked_ids, declined_ids, 
                                   deleted_ids, reported_ids)
    
    if filtered_profiles.empty:
        print("Error: No compatible profiles found after rule-based filtering.")
        return

    # Encode user profile
    user_features = encode_user_profile(user_profile, label_encoders, tfidf)

    # Predict compatibility
    filtered_profiles = predict_compatibility(model, scaler, user_profile['userId'], 
                                            filtered_profiles, X_features, 
                                            user_to_idx, profile_to_idx)
    
    # Top 5 matches
    top_matches = filtered_profiles.sort_values('final_score', ascending=False).head(5)
    
    # Save recommendations
    save_recommendations(top_matches, output_dir="data")

    # Display results
    print("\nTop Compatible Profiles:")
    for _, row in top_matches.iterrows():
        print(f"Profile ID: {row['__id__']}")
        print(f"User Name: {row['userName']}")
        print(f"Age: {row['age']}")
        print(f"Country: {row['country']}")
        print(f"About Me: {row['aboutMe']}")
        print(f"Compatibility Score: {row['final_score']:.2f}")
        print("Reasons:")
        if row['country_match']:
            print("- Matches your country")
        if row['language_match']:
            print("- Matches your language")
        if row['goal_match']:
            print("- Matches your relationship goals")
        if row['ml_score'] > 0.5:
            print("- High AI-predicted compatibility")
        if row['subscribed_score'] > 0:
            print("- Subscribed user")
        if 'soccer' in row['aboutMe'].lower():
            print("- Soccer enthusiast (Africa Soccer Kings compatible)")
        print("---")

if __name__ == "__main__":
    main()