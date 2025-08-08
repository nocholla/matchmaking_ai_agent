import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import time

def train_model(interaction_matrix, X_features):
    """
    Train Gradient Boosting Regressor for compatibility prediction.
    Returns: trained model, scaler
    """
    start_time = time.time()
    num_users, num_items = interaction_matrix.shape
    rows, cols = interaction_matrix.nonzero()
    values = interaction_matrix.data / 2.0  # Normalize to [0, 1]
    
    # Prepare features
    X_train = []
    for user_idx, item_idx in zip(rows, cols):
        features = X_features[item_idx].toarray().flatten()
        X_train.append(np.concatenate([[user_idx, item_idx], features]))
    X_train = np.array(X_train)
    
    # Labels
    y_train = values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_scaled, y_train)
    
    st.write(f"Model Training Time: {time.time() - start_time:.2f} seconds")
    return model, scaler

def predict_compatibility(model, scaler, user_id, filtered_profiles, X_features, user_to_idx, profile_to_idx):
    """
    Predict compatibility scores for filtered profiles.
    Returns: filtered_profiles with ml_score and final_score
    """
    item_indices = [profile_to_idx[pid] for pid in filtered_profiles['__id__'] if pid in profile_to_idx]
    if not item_indices:
        st.error("No valid profiles for ML prediction.")
        return filtered_profiles
    
    user_indices = np.full(len(item_indices), user_to_idx.get(user_id, 0))
    X_pred = [np.concatenate([[u, i], X_features[i].toarray().flatten()]) 
              for u, i in zip(user_indices, item_indices)]
    X_pred = scaler.transform(X_pred)
    scores = model.predict(X_pred)
    
    filtered_profiles['ml_score'] = 0.0
    profile_ids = filtered_profiles['__id__'].values
    for idx, score in zip(item_indices, scores):
        filtered_profiles.loc[filtered_profiles['__id__'] == profile_ids[idx], 'ml_score'] = score
    
    filtered_profiles['final_score'] = (filtered_profiles['ml_score'] * 0.7 + 
                                       filtered_profiles['country_match'].astype(int) * 0.1 + 
                                       filtered_profiles['language_match'].astype(int) * 0.1 + 
                                       filtered_profiles['goal_match'].astype(int) * 0.1)
    
    return filtered_profiles