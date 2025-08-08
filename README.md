🤖 Matchmaking AI Agent
Got it — here’s your **updated README-style description** with **icons** so it reads more like a modern AI project page while keeping your Matchmaking AI Agent’s technical depth.

---

# 🤖 **Matchmaking AI Agent**

An intelligent matchmaking system that combines **rule-based filtering**, **machine learning**, and **text analysis** to recommend the most compatible profiles for users of Africa Love Match.

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
│── data_loader.py
│── preprocessing.py
│── train.py
│── inference.py
│── ui/
│   └── streamlit_app.py
│── models/
│   ├── matchmaking_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoders.pkl
│── data/
│   ├── Profiles.csv
│   ├── LikedUsers.csv
│   └── ...
│── requirements.txt
│── README.md
```

<img width="755" height="724" alt="image" src="https://github.com/user-attachments/assets/2d2bc93f-0cc8-4b4a-a4d9-3ef8c33698ea" />

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

# Or run via Streamlit
streamlit run ui/streamlit_app.py
```

---

<img width="755" height="724" alt="image" src="https://github.com/user-attachments/assets/2d2bc93f-0cc8-4b4a-a4d9-3ef8c33698ea" />


