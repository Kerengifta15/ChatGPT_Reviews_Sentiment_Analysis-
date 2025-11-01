import pandas as pd
import numpy as np
import re
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def load_and_prep(path: str):
    df = pd.read_excel(path)
    # Basic cleaning
    df.columns = [c.strip().lower() for c in df.columns]
    # ensure necessary columns
    if 'review' not in df.columns:
        raise ValueError('Dataset must contain `review` column')
    df['review'] = df['review'].fillna('')
    df['clean_review'] = df['review'].apply(clean_text)
    df['review_length'] = df['review'].astype(str).apply(len)
    # Map rating to sentiment label
    if 'rating' in df.columns:
        def map_sent(r):
            try:
                r = float(r)
            except:
                return 'neutral'
            if r <= 2:
                return 'negative'
            elif r == 3:
                return 'neutral'
            else:
                return 'positive'
        df['sentiment'] = df['rating'].apply(map_sent)
    else:
        # if no rating, sentiment will be Unknown — user may label manually
        df['sentiment'] = 'unknown'

    # Drop duplicates
    df = df.drop_duplicates(subset=['review'])
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input xlsx file')
    parser.add_argument('--output', required=True, help='Path to save cleaned csv')
    args = parser.parse_args()
    df = load_and_prep(args.input)
    df.to_csv(args.output, index=False)
    print('Saved cleaned file to', args.output)
