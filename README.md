🤖 Matchmaking AI Agent

What it does:
Suggests high-compatibility profiles based on data

Tech:
Custom rule engine + ML

data_loader.py: 
Loads CSVs from data/.

preprocessing.py: 
Handles feature engineering (LabelEncoder, TF-IDF, interaction matrix).

recommender.py: 
Trains and predicts with Gradient Boosting Regressor.

agent.py: 
Implements rule-based filtering and user profile encoding.

utils.py: 
Saves models and recommendations.

streamlit_app.py: 
Streamlit UI with caching.

<img width="791" height="733" alt="image" src="https://github.com/user-attachments/assets/182d9a80-1060-416c-aebc-21bae1b934d4" />
