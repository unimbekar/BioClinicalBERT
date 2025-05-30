from transformers import AutoTokenizer
from src import config, preprocess, zero_shot_labeler, dataset, trainer

def main():
    print("\n🔍 Loading and preprocessing data...")
    df = preprocess.load_data("data/clinical_notes.csv")

    print("\n🧠 Generating weak labels with zero-shot classifier...")
    classifier = zero_shot_labeler.get_zero_shot_pipeline()
    df = zero_shot_labeler.apply_labeling(df, classifier)

    print("\n📦 Preparing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = dataset.prepare_dataset(df, tokenizer, config.LABELS)

    print("\n🚀 Training model...")
    trainer.train_model(train_dataset, output_dir="./bioclinicalbert_multilabel")

if __name__ == "__main__":
    main()
