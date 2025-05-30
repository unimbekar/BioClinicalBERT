from src.preprocess import preprocess_notes

def test_preprocess():
    assert preprocess_notes("Patient\n\nhas cancer.  ") == "Patient has cancer."
