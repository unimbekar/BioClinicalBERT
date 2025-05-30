import torch
from src.dataset import ClinicalDataset

def test_dataset():
    texts = ["Patient has diabetes"]
    labels = [[1, 0, 0, 0, 0]]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    dataset = ClinicalDataset(texts, labels, tokenizer)
    assert len(dataset) == 1
    item = dataset[0]
    assert "input_ids" in item and "labels" in item
