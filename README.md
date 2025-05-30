## 🧬 Biomedical Multi-label Classification with Bio_ClinicalBERT

This project fine-tunes Bio_ClinicalBERT for multi-label classification of unstructured medical notes using NLP-based weak supervision.

---

### 🔍 Use Case
Automatically tag medical narratives (clinical notes, encounter summaries, etc.) with relevant health indicators (e.g., `diabetes`, `mental_health`) using domain-specific transformers.

---

### 🧪 Features
- **Bio_ClinicalBERT** fine-tuning for real-time inference.
- **Zero-shot classification** to label data using `facebook/bart-large-mnli`.
- Modular code with unit tests.

---

### 📁 Project Structure
```
bioclinicalbert_project/
├── README.md
├── data/
│   └── clinical_notes.csv            # Input CSV with a 'text' column
├── src/
│   ├── config.py                     # Configuration: labels, thresholds, models
│   ├── preprocess.py                 # Cleans and loads the data
│   ├── zero_shot_labeler.py         # Zero-shot NLP labeling pipeline
│   ├── dataset.py                   # PyTorch dataset wrapper
│   ├── trainer.py                   # Fine-tunes Bio_ClinicalBERT
│   └── main.py                      # End-to-end runner
└── tests/
    ├── test_preprocess.py
    ├── test_labeling.py
    └── test_training.py
```

---

### ⚙️ Setup
```bash
pip install transformers datasets scikit-learn sentence-transformers
```

---

### 🚀 Run Pipeline
```bash
cd bioclinicalbert_project
python src/main.py
```

Make sure your data file `clinical_notes.csv` is in the `data/` folder and includes a `text` column.

---

### 🧪 Run Tests
```bash
pytest tests/
```

---

### 🧠 Notes
- **Labeling Strategy**: Zero-shot classification from `facebook/bart-large-mnli` is used to weakly annotate the dataset.
- **Fine-tuning Model**: `Bio_ClinicalBERT` (by Emily Alsentzer).
- **Multi-label Output**: Trained using sigmoid activation and binary cross-entropy.

---

### 📌 Future Work
- Use contrastive learning or sentence embeddings for self-supervised pre-labeling.
- Evaluate performance across clinical subtypes (e.g., discharge summaries vs. intake).

---

### 🧑‍⚕️ Authors
Created for real-world healthcare NLP applications where labeled data is scarce but unstructured notes are abundant.

---

### 📄 License
MIT License
