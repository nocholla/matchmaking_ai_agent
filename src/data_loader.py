import pandas as pd
import os
import streamlit as st
import time

def load_data(data_dir="data"):
    """
    Load and merge CSV datasets.
    Returns: profiles_df, liked_df, matched_df, blocked_ids, declined_ids, deleted_ids, reported_ids
    """
    start_time = time.time()  # Record start time
    required_cols = ['__id__', 'userId', 'userName', 'age', 'country', 'language', 
                     'aboutMe', 'sex', 'seeking', 'relationshipGoals', 'subscribed',
                     'subscribedEliteOne', 'subscribedEliteThree', 'subscribedEliteSix',
                     'subscribedEliteTwelve']
    
    try:
        profiles = pd.read_csv(
            os.path.join(data_dir, "Profiles.csv"),
            usecols=[c for c in required_cols if c in pd.read_csv(os.path.join(data_dir, "Profiles.csv"), nrows=1).columns]
        )
        liked = pd.read_csv(os.path.join(data_dir, "LikedUsers.csv"), usecols=['userId', '__id__'])
        matched = pd.read_csv(os.path.join(data_dir, "MatchedUsers.csv"), usecols=['userId', '__id__'])
        blocked = pd.read_csv(os.path.join(data_dir, "BlockedUsers.csv"), usecols=['__id__'])
        declined = pd.read_csv(os.path.join(data_dir, "DeclinedUsers.csv"), usecols=['__id__'])
        deleted = pd.read_csv(os.path.join(data_dir, "DeletedUsers.csv"), usecols=['__id__'])
        reported = pd.read_csv(os.path.join(data_dir, "ReportedUsers.csv"), usecols=['__id__'])
    except FileNotFoundError as e:
        st.error(f"CSV file not found: {e}")
        return None, None, None, None, None, None, None
    
    profiles = profiles.drop_duplicates().fillna("unknown")
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    st.write(f"Data Loading Time: {elapsed_time:.2f} seconds")
    return (profiles, liked, matched, 
            blocked['__id__'].tolist(), declined['__id__'].tolist(), 
            deleted['__id__'].tolist(), reported['__id__'].tolist())