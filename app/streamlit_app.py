# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
from collections import Counter
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.figure_factory as ff

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load("models/model_lr.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit Configuration
st.set_page_config(page_title="AI Sentiment Analysis Dashboard", layout="wide")
st.title("💬 AI-Powered Sentiment Analysis for ChatGPT Reviews")
st.markdown("### 📊 Analyze, Visualize, and Evaluate Sentiment Intelligence")

# Sidebar - File Upload
st.sidebar.header("⚙️ Upload Options")
uploaded_file = st.sidebar.file_uploader("Upload your cleaned CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/cleaned_chatgpt_reviews.csv")

df.columns = [c.strip().lower() for c in df.columns]

# Prediction
if "clean_review" not in df.columns:
    df["clean_review"] = df["review"].astype(str)

if "sentiment" not in df.columns:
    X_vec = vectorizer.transform(df["clean_review"])
    df["sentiment"] = model.predict(X_vec)

# --- Sidebar Filters and Quick Stats ---
st.sidebar.markdown("---")
st.sidebar.header("📈 Quick Stats")

if 'rating' in df.columns:
    avg_rating = round(df['rating'].mean(), 2)
    st.sidebar.metric("Average Rating ⭐", avg_rating)
    st.sidebar.metric("Total Reviews", len(df))
    st.sidebar.metric("Unique Users", df['username'].nunique() if 'username' in df.columns else "N/A")

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Reviews")

# Apply filters before visualizations
if 'rating' in df.columns:
    min_r = int(df['rating'].min())
    max_r = int(df['rating'].max())
    rating_filter = st.sidebar.slider("Filter by Rating", min_r, max_r, (min_r, max_r))
    df = df[(df['rating'] >= rating_filter[0]) & (df['rating'] <= rating_filter[1])]

if 'sentiment' in df.columns:
    selected_sentiments = st.sidebar.multiselect(
        "Select Sentiment",
        options=df['sentiment'].unique(),
        default=df['sentiment'].unique()
    )
    df = df[df['sentiment'].isin(selected_sentiments)]

# Dataset Overview
st.subheader("📘 Dataset Overview")
st.dataframe(df.head(10))
st.write(f"Total Reviews Loaded: **{len(df)}**")

# EDA Tabs
st.header("🔍 Exploratory Data Analysis (EDA)")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "1️⃣ Rating Distribution",
    "2️⃣ Helpful Reviews",
    "3️⃣ Keywords (Pos vs Neg)",
    "4️⃣ Ratings Over Time",
    "5️⃣ Ratings by Location",
    "6️⃣ Platform Comparison",
    "7️⃣ Verified vs Non-Verified",
    "8️⃣ Review Lengths",
    "9️⃣ 1-Star Keywords",
    "🔟 Version Comparison",
    "📈 Model Evaluation Metrics",
])

# --- 1. Rating Distribution ---
with tab1:
    if "rating" in df.columns:
        fig = px.histogram(df, x="rating", nbins=5, color="sentiment",
                           title="Distribution of Review Ratings (1–5 Stars)")
        st.plotly_chart(fig, use_container_width=True)

        # Pie chart of sentiment distribution
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts = sentiment_counts.reset_index()
        fig_pie = px.pie(sentiment_counts, names="sentiment", values="count",
                 title="Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("⚠️ 'rating' column not found.")

# --- 2. Helpful Reviews ---
with tab2:
    if "helpful_votes" in df.columns:
        df["helpful_category"] = df["helpful_votes"].apply(lambda x: "Helpful" if x >= 10 else "Not Helpful")
        helpful_counts = df["helpful_category"].value_counts().reset_index()
        helpful_counts.columns = ["Category", "Count"]
        fig = px.pie(helpful_counts, names="Category", values="Count", title="Helpful vs Non-Helpful Reviews")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Shows what percentage of reviews were marked as helpful by other users.")
    else:
        st.warning("⚠️ 'helpful_votes' column not found in dataset.")

# --- 3. Common Keywords (Positive vs Negative) ---
with tab3:
    st.subheader("🔠 Common Keywords in Positive vs Negative Reviews")

    def clean_and_extract(texts):
        text = " ".join(texts).lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)
        stopwords = {"the", "and", "for", "with", "that", "this", "have", "from", "not", "was",
                     "are", "but", "you", "your", "had", "has", "been", "they", "were", "will"}
        words = [w for w in words if w not in stopwords]
        return Counter(words).most_common(10)

    if "rating" in df.columns and "clean_review" in df.columns:
        pos_reviews = df[df["rating"] >= 4]["clean_review"].astype(str)
        neg_reviews = df[df["rating"] <= 2]["clean_review"].astype(str)

        if not pos_reviews.empty and not neg_reviews.empty:
            pos_words = clean_and_extract(pos_reviews)
            neg_words = clean_and_extract(neg_reviews)

            pos_df = pd.DataFrame(pos_words, columns=["word", "count"])
            neg_df = pd.DataFrame(neg_words, columns=["word", "count"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 😊 Positive Reviews (4–5 Stars)")
                fig1 = px.bar(pos_df, x="word", y="count", color="count",
                              title="Top 10 Keywords in Positive Reviews")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.markdown("### 😠 Negative Reviews (1–2 Stars)")
                fig2 = px.bar(neg_df, x="word", y="count", color="count",
                              title="Top 10 Keywords in Negative Reviews")
                st.plotly_chart(fig2, use_container_width=True)

            st.info("Insight: Discover what users love or complain about based on frequent keywords.")
        else:
            st.warning("Not enough reviews in 4–5 or 1–2 rating categories.")
    else:
        st.warning("⚠️ Required columns ('rating', 'clean_review') not found.")

# --- 4. Ratings Over Time ---
with tab4:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        avg_rating_over_time = df.groupby("date")["rating"].mean().reset_index()
        fig = px.line(avg_rating_over_time, x="date", y="rating",
                      title="Average Rating Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Tracks how user satisfaction changes over time — dips may indicate poor updates.")
    else:
        st.warning("⚠️ 'date' column not found in dataset.")

# --- 5. Ratings by Location ---
with tab5:
    if "location" in df.columns:
        location_avg = df.groupby("location")["rating"].mean().reset_index().sort_values("rating", ascending=False).head(10)
        fig = px.bar(location_avg, x="location", y="rating", color="location",
                     title="Top 10 Locations by Average Rating")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Identifies regions with highest and lowest satisfaction levels.")
    else:
        st.warning("⚠️ 'location' column not found in dataset.")

# --- 6. Platform Comparison ---
with tab6:
    if "platform" in df.columns:
        platform_avg = df.groupby("platform")["rating"].mean().reset_index()
        fig = px.bar(platform_avg, x="platform", y="rating", color="platform",
                     title="Average Ratings by Platform (Web vs App)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Helps understand which platform provides a better experience to users.")
    else:
        st.warning("⚠️ 'platform' column not found in dataset.")

# --- 7. Verified vs Non-Verified ---
with tab7:
    if "verified_purchase" in df.columns:
        verified_avg = df.groupby("verified_purchase")["rating"].mean().reset_index()
        fig = px.bar(verified_avg, x="verified_purchase", y="rating", color="verified_purchase",
                     title="Verified vs Non-Verified User Ratings", text="rating")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Indicates whether verified users tend to rate higher or lower than others.")
    else:
        st.warning("⚠️ 'verified_purchase' column not found in dataset.")

# --- 8. Review Length by Rating ---
with tab8:
    if "review_length" in df.columns and "rating" in df.columns:
        df["review_length"] = pd.to_numeric(df["review_length"], errors="coerce")
        fig = px.box(df, x="rating", y="review_length",
                     title="Review Length by Rating Category")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Shows whether longer reviews are more common among happy or unhappy users.")
    else:
        st.warning("⚠️ 'review_length' or 'rating' column not found.")

# --- 9. Top Words in 1-Star Reviews ---
with tab9:
    st.subheader("📉 Common Complaint Keywords in 1-Star Reviews")

    if "rating" in df.columns and "clean_review" in df.columns:
        one_star = df[df["rating"] == 1]["clean_review"].astype(str)
        if not one_star.empty:
            text = " ".join(one_star).lower()
            words = re.findall(r'\b[a-z]{3,}\b', text)
            stopwords = {"the", "and", "for", "with", "that", "this", "have", "from", "not",
                         "was", "are", "but", "you", "your", "had", "has", "been",
                         "they", "were", "will"}
            words = [w for w in words if w not in stopwords]
            counts = Counter(words).most_common(15)
            word_df = pd.DataFrame(counts, columns=["word", "count"])

            fig = px.bar(word_df, x="word", y="count", color="count",
                         title="Most Frequent Complaint Keywords in 1-Star Reviews")
            st.plotly_chart(fig, use_container_width=True)
            st.info("Insight: Identifies frequent complaints such as 'bug', 'slow', 'error', 'problem'.")
        else:
            st.warning("No 1-star reviews found in dataset.")
    else:
        st.warning("⚠️ Required columns ('rating', 'clean_review') not found.")

# --- 10. Average Rating by ChatGPT Version ---
with tab10:
    if "version" in df.columns:
        version_avg = df.groupby("version")["rating"].mean().reset_index()
        fig = px.bar(version_avg, x="version", y="rating", color="version",
                     title="Average Rating by ChatGPT Version")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Insight: Evaluates improvement or regression across ChatGPT versions.")
    else:
        st.warning("⚠️ 'version' column not found in dataset.")

# --- 11. Model Evaluation Metrics ---
with tab11:
    st.subheader("📈 Model Performance Evaluation")
    try:
        if "sentiment" in df.columns:
            X_vec = vectorizer.transform(df["clean_review"].astype(str))
            df["predicted_sentiment"] = model.predict(X_vec)

            y_true = df["sentiment"]
            y_pred = df["predicted_sentiment"]

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            # Display metrics as cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("✅ Accuracy", f"{accuracy:.3f}")
            col2.metric("🎯 Precision", f"{precision:.3f}")
            col3.metric("📍 Recall", f"{recall:.3f}")
            col4.metric("💡 F1 Score", f"{f1:.3f}")

            # Metrics Table
            st.markdown("### 📊 Evaluation Metrics Table")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Value": [accuracy, precision, recall, f1]
            })
            st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))

            # Confusion Matrix
            st.markdown("### 🔲 Confusion Matrix")
            labels = sorted(df["sentiment"].unique())
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            normalize = st.checkbox("Show Normalized Confusion Matrix", value=False)
            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, None]

            fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale="Blues", showscale=True)
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)

            # Download predicted results
            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions as CSV", data=csv_download, file_name="sentiment_predictions.csv")

        else:
            st.warning("⚠️ 'sentiment' column not found.")
    except Exception as e:
        st.error(f"Error while computing metrics: {e}")

# Custom Review Sentiment Prediction
st.header("💬 Try Your Own Review")
user_input = st.text_area("Type your review below and click **Analyze Sentiment** 👇")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        st.success(f"Predicted Sentiment: **{pred.upper()}** 🎯")
    else:
        st.warning("Please type something before analyzing.")

# Footer
st.markdown("---")
st.caption("Developed by **Keren Gifta A** | 🌐 AI Sentiment Dashboard | © 2025")
