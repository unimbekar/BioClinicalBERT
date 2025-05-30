import torch
from sklearn.preprocessing import MultiLabelBinarizer

class ClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

def prepare_dataset(df, tokenizer, labels):
    mlb = MultiLabelBinarizer(classes=labels)
    Y = mlb.fit_transform(df['labels'])
    return ClinicalDataset(df['text'].tolist(), Y, tokenizer)
