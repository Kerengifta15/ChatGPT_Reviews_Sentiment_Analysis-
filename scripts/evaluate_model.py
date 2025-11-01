import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(csv_path, model_path, vec_path):
    # Load dataset
    df = pd.read_csv(csv_path)
    df = df[df['sentiment'].isin(['positive', 'neutral', 'negative'])]

    X = df['clean_review']
    y = df['sentiment']

    # Load model + vectorizer
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)

    # Transform data
    X_vec = vectorizer.transform(X)

    # Predict
    y_pred = model.predict(X_vec)

    # Print report
    print("✅ Evaluation Results")
    print("=" * 40)
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--vec", required=True, help="Path to saved vectorizer file")
    args = parser.parse_args()

    evaluate(args.input, args.model, args.vec)
