import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import time
import logging
from src.data_loader import load_data, load_config
from src.preprocessing import preprocess_data
from src.recommender import train_model, predict_compatibility, load_model_and_encoders
from src.agent import apply_rules, encode_user_profile
from src.utils import save_models, save_recommendations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource
def cached_load_data(data_dir):
    return load_data(data_dir=data_dir)

@st.cache_resource
def cached_preprocess_data(profiles, liked, matched):
    return preprocess_data(profiles, liked, matched)

@st.cache_resource
def cached_train_model(_interaction_matrix, _X_features):
    return train_model(_interaction_matrix, _X_features)

def main():
    st.title("Matchmaking AI Agent 🤖")
    st.markdown("""
    Enter a user ID or profile details to get high-compatibility profile suggestions.
    """)

    # Load config
    config = load_config()
    data_dir = config['data']['data_dir']
    models_dir = config['model']['models_dir']

    # Initialize variables
    profiles = None
    liked = None
    matched = None
    blocked_ids = None
    declined_ids = None
    deleted_ids = None
    reported_ids = None
    interaction_matrix = None
    X_features = None
    user_to_idx = None
    profile_to_idx = None
    label_encoders = None
    tfidf = None
    model = None
    scaler = None

    # Check for existing models
    model_files = ["matchmaking_model.pkl", "scaler.pkl", "label_encoders.pkl", 
                   "tfidf_vectorizer.pkl", "user_to_idx.pkl", "profile_to_idx.pkl"]
    if all(os.path.exists(os.path.join(models_dir, f)) for f in model_files):
        model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx = load_model_and_encoders(models_dir)
        if model is None:
            st.error("Failed to load models. Please ensure model files are valid.")
            st.stop()
        # Load data for profiles and other necessary data
        start_time = time.time()
        (profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids) = cached_load_data(data_dir)
        if profiles is None:
            st.error("Failed to load data. Please check the data directory and CSV files.")
            st.stop()
        # Preprocess data to get X_features
        (profiles, interaction_matrix, X_features, user_to_idx, profile_to_idx, 
         label_encoders, tfidf) = cached_preprocess_data(profiles, liked, matched)
    else:
        # Load data
        start_time = time.time()
        (profiles, liked, matched, blocked_ids, declined_ids, deleted_ids, reported_ids) = cached_load_data(data_dir)
        if profiles is None:
            st.error("Failed to load data. Please check the data directory and CSV files.")
            st.stop()

        # Preprocess data
        (profiles, interaction_matrix, X_features, user_to_idx, profile_to_idx, 
         label_encoders, tfidf) = cached_preprocess_data(profiles, liked, matched)

        # Train model
        model, scaler = cached_train_model(interaction_matrix, X_features)

        # Save models
        save_models(model, scaler, label_encoders, tfidf, user_to_idx, profile_to_idx, models_dir)
        logger.info(f"Model training and saving completed in {time.time() - start_time:.2f} seconds")

    # User input
    user_id_input = st.text_input("User ID", "user123")
    age_input = st.slider("Age", 18, 70, 25)
    sex_input = st.selectbox("Sex", ["Female", "Male", "unknown"])
    seeking_input = st.selectbox("Seeking", ["Male", "Female", "unknown"])
    country_input = st.text_input("Country", "Kenya")
    language_input = st.text_input("Language", "Swahili")
    relationship_goals_input = st.text_input("Relationship Goals", "Long-term")
    about_me_input = st.text_area("About Me", "Looking for true love and enjoy soccer!")

    if st.button("Find Matches"):
        user_profile = {
            'userId': user_id_input,
            'age': age_input,
            'sex': sex_input,
            'seeking': seeking_input,
            'country': country_input,
            'language': language_input,
            'relationshipGoals': relationship_goals_input,
            'aboutMe': about_me_input
        }
        
        # Apply rules
        filtered_profiles = apply_rules(profiles, user_profile, blocked_ids, declined_ids, 
                                       deleted_ids, reported_ids)
        
        if filtered_profiles.empty:
            st.error("No compatible profiles found after rule-based filtering.")
        else:
            # Encode user profile
            user_features = encode_user_profile(user_profile, label_encoders, tfidf)
            
            # Predict compatibility
            filtered_profiles = predict_compatibility(model, scaler, user_id_input, 
                                                    filtered_profiles, X_features, 
                                                    user_to_idx, profile_to_idx)
            
            # Top 5 matches
            top_matches = filtered_profiles.sort_values('final_score', ascending=False).head(5)
            
            # Save recommendations
            save_recommendations(top_matches, output_dir=data_dir)
            
            st.subheader("Top Compatible Profiles:")
            for _, row in top_matches.iterrows():
                st.write(f"Profile ID: {row['__id__']}")
                st.write(f"User Name: {row['userName']}")
                st.write(f"Age: {row['age']}")
                st.write(f"Country: {row['country']}")
                st.write(f"About Me: {row['aboutMe']}")
                st.write(f"Compatibility Score: {row['final_score']:.2f}")
                st.write("Reasons:")
                if row['country_match']:
                    st.write("- Matches your country")
                if row['language_match']:
                    st.write("- Matches your language")
                if row['goal_match']:
                    st.write("- Matches your relationship goals")
                if row['ml_score'] > 0.5:
                    st.write("- High AI-predicted compatibility")
                if row['subscribed_score'] > 0:
                    st.write("- Subscribed user")
                if 'soccer' in row['aboutMe'].lower():
                    st.write("- Soccer enthusiast (Africa Soccer Kings compatible)")
                st.write("---")

if __name__ == "__main__":
    main()