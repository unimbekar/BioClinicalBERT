from transformers import pipeline
from .config import LABELS, ZERO_SHOT_MODEL, THRESHOLD

def get_zero_shot_pipeline():
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)

def label_with_zero_shot(text, classifier):
    result = classifier(text, LABELS, multi_label=True)
    return [label for label, score in zip(result['labels'], result['scores']) if score > THRESHOLD]

def apply_labeling(df, classifier):
    df['labels'] = df['text'].apply(lambda x: label_with_zero_shot(x, classifier))
    return df
