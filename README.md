# Phishing Email Classification Using Fine-tuned DistilBERT

Phishing remains a critical threat to cybersecurity. This project develops a high-performance email classifier capable of distinguishing between legitimate ("ham") and malicious ("phishing") emails. By leveraging the power of pretrained transformer models, the system captures linguistic and contextual patterns specific to fraudulent communications.

## Key Features

**Transformer-based Classification**: Utilizes the `distilbert-base-uncased` model for efficient and accurate NLP performance.

**Dataset**: Built from a combined collection of ~82,000 emails from six major sources (Enron, Ling, Nazario, Nigerian Fraud, SpamAssassin, and CEAS).

**Fine-tuning Strategy**: Implements partial layer unfreezing, keeping the initial five layers frozen while fine-tuning the top layers for domain-specific accuracy.

**High Performance**: Achieves over **99% validation accuracy** with minimal loss.

## Dataset Summary

The dataset is a composite of legitimate and scam emails, pre-processed to remove duplicates and combine subject lines, bodies, and sender metadata into a single text input.

* **Class Balance**: Roughly 39,600 legitimate emails and 42,900 phishing emails.
* **Sequence Preparation**: Emails are tokenized with a maximum sequence length of 256 tokens.

## Model Architecture & Training

The classifier architecture consists of the DistilBERT core followed by a pre-classifier (768-dim) and a final classification head.

| Component | Training Status |
| --- | --- |
| Embeddings | Frozen |
| Transformer Blocks 0-3 | Frozen |
| Transformer Blocks 4-5 | <br>**Fine-tuned** |
| Pre-classifier | <br>**Fine-tuned** |
| Classifier Head | <br>**Fine-tuned** |

**Hyperparameters:**

* **Framework**: PyTorch & Hugging Face Transformers
* **Optimizer**: AdamW
* **Initial Learning Rate**: 5e-5 

**Callbacks**: Early Stopping & `ReduceLROnPlateau` scheduler 

## Results

The model demonstrates exceptional generalization, converging effectively within 5 to 10 epochs.

**Validation Accuracy**: 99.39% 

**Validation Loss**: 0.0209 

**Performance Metrics**: Rare false positives and robust detection across both email classes.

## Tech Stack

* **Language**: Python
* **Libraries**: PyTorch, Hugging Face `transformers`, Scikit-learn, Pandas, Matplotlib
* **Source Data**: Kaggle (Kagglehub)

## Contributors

* Ade Aiho
* Heta Hartzell
* Mika Laakkonen
* Jonne Roponen
