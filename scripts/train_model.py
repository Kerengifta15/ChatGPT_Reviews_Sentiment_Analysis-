import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os

def train_tfidf_lr(csv_path, model_out="models/model_lr.joblib", vec_out="models/vectorizer.joblib"):
    df = pd.read_csv(csv_path)

    # Keep only needed columns
    df = df[df["sentiment"].isin(["positive", "neutral", "negative"])]

    X = df["clean_review"]
    y = df["sentiment"]

    # Convert text to numeric TF-IDF features
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    # Balance dataset
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_vec, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model & vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, model_out)
    joblib.dump(vectorizer, vec_out)

    print(f"✅ Model saved to: {model_out}")
    print(f"✅ Vectorizer saved to: {vec_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_out", default="models/model_lr.joblib")
    parser.add_argument("--vec_out", default="models/vectorizer.joblib")
    args = parser.parse_args()

    train_tfidf_lr(args.input, args.model_out, args.vec_out)