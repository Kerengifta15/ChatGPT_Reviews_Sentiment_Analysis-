# 🤖 AI Sentiment Analysis of ChatGPT Reviews  

An interactive **Streamlit web app** that performs **AI-powered sentiment analysis** on ChatGPT user reviews using **Machine Learning and Natural Language Processing (NLP)**.  
The project analyzes user opinions, identifies trends, and provides data-driven insights through intuitive visualizations.  

---

## 🧩 Features

- 🔍 **Sentiment Prediction** — Classifies reviews as Positive, Negative, or Neutral  
- 📊 **Dynamic Dashboard** — Interactive charts and plots built using Plotly  
- ⚙️ **Smart Sidebar Controls** — Search, filter, and explore reviews easily  
- 🧠 **Model Evaluation Metrics** — Displays accuracy, precision, recall, and F1-score  
- 💬 **Keyword & Trend Analysis** — Understand what users talk about the most  
- 🧩 **Custom Feature Controls** — Explore keyword search, review explorer, and auto tagging  

---

## 🧠 Model Details

This project uses a **Machine Learning–based NLP pipeline** built with **Scikit-learn**.  
- **Vectorizer:** TF-IDF (Term Frequency–Inverse Document Frequency)  
- **Classifier:** Logistic Regression (trained to predict sentiment)  

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression())
])
