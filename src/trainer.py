import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score
from .config import MODEL_NAME

def compute_metrics(pred):
    logits, labels = pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_precision": precision_score(labels, preds, average="micro"),
        "micro_recall": recall_score(labels, preds, average="micro")
    }

def train_model(train_dataset, output_dir="./model"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(train_dataset.labels[0]),
        problem_type="multi_label_classification"
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(output_dir)
