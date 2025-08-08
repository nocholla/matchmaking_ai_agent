# 🤖 **Matchmaking AI Agent**

An intelligent matchmaking system that combines **rule-based filtering**, **machine learning**, and **text analysis** to recommend the most compatible profiles for users.

---

## ✨ **Features**

* 📝 **User Profile Input**
  Users can input details like **age**, **sex**, **seeking preference**, **country**, **language**, **relationship goals**, and a personal bio.

* 🛡 **Rule-Based Filtering**
  Filters profiles based on compatibility rules:

  * Matching sex/seeking preferences
  * Age range ±5 years
  * Excluding blocked/declined users

* 📈 **Machine Learning Predictions**
  Uses a **Gradient Boosting Regressor** to predict compatibility scores, enhanced with **TF-IDF** vectorization of user bios.

* ⚽ **Soccer Enthusiast Detection**
  Identifies soccer fans for compatibility with **Africa Soccer Kings**, based on keywords in the `aboutMe` field.

* ⚡ **Caching for Performance**
  **Streamlit caching** (`@st.cache_resource`) optimizes data loading, preprocessing, and model training.

* 💾 **Model Persistence**
  Saves trained models and encoders for reuse:

  ```
  matchmaking_model.pkl
  tfidf_vectorizer.pkl
  label_encoders.pkl
  ```

* 📊 **Recommendation Output**
  Saves **top matches** to `recommendations.csv` with **reasons for compatibility**.

---

## 🛠 **Tech Stack**

* **Programming Language:** 🐍 Python 3.13
* **Web Framework:** 🌐 Streamlit (interactive UI)
* **Machine Learning:**

  * `scikit-learn` → GradientBoostingRegressor, StandardScaler, LabelEncoder, TfidfVectorizer
  * `pandas` → CSV handling and data manipulation
  * `numpy` → Numerical computing
  * `scipy` → Sparse matrix handling
* **Data Storage:** 📂 CSV files (`Profiles.csv`, `LikedUsers.csv`, etc.)
* **Model Serialization:** `joblib` for model saving/loading
* **Dependencies:** Managed with `requirements.txt`

---

## 🚀 **Extras**

* ⚡ **Performance Optimization** — Streamlit resource caching for fast reloads
* 🔍 **Keyword-Based Scoring** — Boosts compatibility for profiles mentioning "love", "soccer", "relationship"
* 💎 **Subscription Prioritization** — Premium members (`subscribed`, `subscribedEliteOne`) get score boosts
* 🛠 **Error Handling** — Friendly messages for missing files or invalid inputs
* 🧩 **Extensibility** — Modular design (e.g., `data_loader.py`, `preprocessing.py`) for easy updates and Firebase integration

---

## 📁 **Project Structure**

```
matchmaking_ai_agent/
│
├── data/
│   ├── Profiles.csv
│   ├── LikedUsers.csv
│   ├── ...
│
├── models/
│   ├── matchmaking_model.pkl
│   ├── ...
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── recommender.py
│   ├── agent.py
│   ├── utils.py
│
├── ui/
│   ├── streamlit_app.py
│   ├── cli.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_agent.py
│
├── requirements.txt
├── README.md
└── run.py
```

---

## 📦 **Installation**

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/matchmaking_ai_agent.git
cd matchmaking_ai_agent

# 2️⃣ Create and activate a Python 3.13 virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Ensure data CSVs are in the /data directory
```

---

## ▶ **Run the App**

```bash
# Run via Python
python run.py

python3 run.py

# Or run via Streamlit
streamlit run ui/streamlit_app.py
```

---
## 🤖 **App**

<img width="1703" height="799" alt="1 Matchmaking Agent " src="https://github.com/user-attachments/assets/53fc99ac-b3be-4b9a-a3c8-97e7fd5f1e41" />

<img width="3406" height="1560" alt="image" src="https://github.com/user-attachments/assets/7b80bfd4-2678-46d9-8317-4d45bd89681e" />

---


