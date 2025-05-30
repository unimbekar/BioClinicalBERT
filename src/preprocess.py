import re
import pandas as pd

def preprocess_notes(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str).apply(preprocess_notes)
    return df
