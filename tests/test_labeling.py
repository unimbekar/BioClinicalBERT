from src.zero_shot_labeler import label_with_zero_shot, get_zero_shot_pipeline

classifier = get_zero_shot_pipeline()

def test_label():
    text = "Patient has a history of hypertension and diabetes."
    labels = label_with_zero_shot(text, classifier)
    assert 'diabetes' in labels or 'hypertension' in labels
