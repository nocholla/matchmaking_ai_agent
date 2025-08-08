🤖 Matchmaking AI Agent

Features
1) User Profile Input: Users can input details like age, sex, seeking preference, country, language, relationship goals, and a personal bio.
2) Rule-Based Filtering: Filters profiles based on compatibility rules (e.g., matching sex/seeking preferences, age range ±5 years, excluding blocked/declined users).
3) Machine Learning Predictions: Uses a Gradient Boosting Regressor to predict compatibility scores, enhanced by TF-IDF vectorization of user bios.
4) Soccer Enthusiast Detection: Identifies soccer fans for compatibility with Africa Soccer Kings, based on keywords in the aboutMe field.
5) Caching for Performance: Streamlit caching (@st.cache_resource) optimizes data loading, preprocessing, and model training.
6) Model Persistence: Saves trained models and encoders (e.g., matchmaking_model.pkl, tfidf_vectorizer.pkl) for reuse.
7) Recommendation Output: Saves top matches to recommendations.csv with reasons for compatibility.

Tech Stack
1) Programming Language: Python 3.13
2) Web Framework: Streamlit for the interactive user interface
3) Machine Learning:
- scikit-learn: GradientBoostingRegressor for compatibility prediction, StandardScaler for feature scaling, LabelEncoder for categorical encoding, TfidfVectorizer for text processing
- pandas: Data manipulation and CSV handling
- numpy: Numerical computations and array operations
- scipy: Sparse matrix operations for interaction matrices
4) Data Storage: CSV files (Profiles.csv, LikedUsers.csv, MatchedUsers.csv, etc.) for user data and interactions
5) Model Serialization: joblib for saving/loading trained models and encoders
6) Dependencies: Managed via a virtual environment (see requirements.txt)

Extras
1) Performance Optimization: Uses Streamlit's @st.cache_resource to cache data loading, preprocessing, and model training, reducing runtime for repeated queries.
2) Keyword-Based Scoring: Enhances recommendations by scoring profiles based on keywords like "love," "soccer," and "relationship" in the aboutMe field.
3) Subscription Prioritization: Boosts scores for subscribed users (e.g., subscribed, subscribedEliteOne) to prioritize premium profiles.
4) Error Handling: Robust handling of missing CSV files and invalid user inputs, with user-friendly error messages in the Streamlit UI.
5) Extensibility: Modular design with separate modules (data_loader.py, preprocessing.py, etc.) for easy maintenance and future enhancements (e.g., Firebase integration for real-time data).

Project Structure

<img width="755" height="724" alt="image" src="https://github.com/user-attachments/assets/2d2bc93f-0cc8-4b4a-a4d9-3ef8c33698ea" />

Installation

1) Clone the repository:

git clone https://github.com/<your-username>/matchmaking_ai_agent.git
cd matchmaking_ai_agent

2) Set up a Python 3.13 virtual environment:

python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3) Install dependencies:

pip install -r requirements.txt


4) Ensure the data directory contains the required CSV files (e.g., Profiles.csv).

5) Run the app:

python run.py

Or

python3 run.py

Or 

streamlit run ui/streamlit_app.py



