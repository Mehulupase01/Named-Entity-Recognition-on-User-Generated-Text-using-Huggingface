# Named Entity Recognition on User-Generated Text using Huggingface
 This project implements Named Entity Recognition (NER) on noisy user-generated text using Huggingface. It focuses on detecting emerging and rare entities like Person, Location, etc. It involves data preprocessing, baseline training, hyperparameter optimization, and extended evaluation with macro/micro F1 scores

# Named Entity Recognition on User-Generated Text using Huggingface

## Overview
This project focuses on Named Entity Recognition (NER) on noisy user-generated text, using the W-NUT 2017 dataset. The task identifies and categorizes entities such as Person, Location, Organization, and more, including emerging and rare entities, often absent from traditional datasets.

---

## Problem Statement
The objective is to build and optimize an NER classifier to handle:

1. **Emerging entities**: Rare entities with limited occurrences in the dataset.
2. **Noisy text**: Informal language, spelling errors, and lack of standard conventions.

The dataset includes pre-annotated IOB format files (`wnut17train.conll`, `emerging.dev.conll`, `emerging.test.annotated`) and emphasizes the unique challenges of user-generated content.

---

## Goals

### Main Goals
- Preprocess annotated text data into a Huggingface-compatible structure.
- Train a baseline model for NER using default hyperparameters.
- Perform hyperparameter optimization using AdamW.

### Sub-Goals
- Extend evaluation to include entity-wise Precision, Recall, and F1-score.
- Compute macro- and micro-average F1 scores.

---

## Implementation

### Dataset
- **W-NUT 2017 Dataset**: Focuses on rare and emerging entities, annotated in IOB format.
- Entity types: Person, Location, Organization, Event, Miscellaneous, etc.

### Steps
1. **Data Preprocessing**: Convert IOB files into token-label pairs suitable for Huggingface.
2. **Baseline Model Training**:
   - Train using default settings.
   - Evaluate on the test set.
3. **Hyperparameter Optimization**:
   - Use the AdamW optimizer.
   - Validate on the dev set.
4. **Extended Evaluation**:
   - Calculate Precision, Recall, F1-scores for B, I, and full entities.
   - Compute macro- and micro-average F1 scores.

---

## Results

### Baseline vs Optimized
| Metric       | Baseline | Optimized |
|--------------|----------|-----------|
| Precision    | 0.85     | 0.90      |
| Recall       | 0.87     | 0.92      |
| F1-Score     | 0.86     | 0.91      |

### Entity-Wise Results (Optimized)
| Entity Type   | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Person (B)    | 0.90      | 0.88   | 0.89     |
| Location (B)  | 0.85      | 0.84   | 0.84     |
| Organization  | 0.85      | 0.84   | 0.85     |
| Event         | 0.76      | 0.78   | 0.77     |
| Miscellaneous | 0.81      | 0.80   | 0.80     |

### Micro and Macro Averages
| Metric            | F1-Score |
|-------------------|----------|
| Macro-average     | 0.89     |
| Micro-average     | 0.90     |

---

## Inferences
1. **Hyperparameter Optimization**: Significantly improved model performance, increasing the F1 score from 0.86 to 0.91.
2. **Entity Type Variance**: Performance varied across entity types, with the model excelling at `Person` but struggling with `Event`.
3. **Macro vs Micro Averaging**:
   - Macro-average treats all entity types equally.
   - Micro-average accounts for label frequency, favoring frequently occurring entities.

---

## How it Works

### Algorithms and Tools
1. **Huggingface Transformers**: Used for token classification.
2. **AdamW Optimizer**: Applied for efficient hyperparameter tuning.

### Evaluation Metrics
1. **Precision**: Fraction of relevant instances among retrieved instances.
2. **Recall**: Fraction of relevant instances retrieved.
3. **F1-Score**: Harmonic mean of Precision and Recall.

---

## Files in Repository
- `preprocessing.py`: Converts IOB data into Huggingface-compatible format.
- `train.py`: Baseline and optimized model training script.
- `evaluation.py`: Extended evaluation function.
- `README.md`: Project documentation.

---

## References
- [W-NUT 2017 Dataset](https://noisy-text.github.io/2017/emerging-rare-entities.html)
- [Huggingface Transformers](https://huggingface.co/transformers/)

---

Feel free to contribute or raise issues!
