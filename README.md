# SentimentScope: Transformers for Movie Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![Udacity](https://img.shields.io/badge/Udacity-AI%20Programming%20with%20Python-02B5E2.svg?logo=udacity&logoColor=white)](https://www.udacity.com/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-75.70%25-brightgreen.svg)](#experimental-results)

An end-to-end deep learning project for binary sentiment classification on the large-scale **IMDB Movie Reviews Dataset (50,000 reviews)**. Built for the entertainment platform **CineScope**, this project customizes and trains a transformer neural network using subword tokenization (`bert-base-uncased`), multi-head self-attention, and sequence-level mean pooling.

Developed as part of the **AWS & Udacity AI Programming Nanodegree Program**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features & Architecture](#key-features--architecture)
- [Dataset & Exploratory Analysis](#dataset--exploratory-analysis)
- [Model Architecture Details](#model-architecture-details)
- [Training & Validation Dynamics](#training--validation-dynamics)
- [Experimental Results](#experimental-results)
- [Interactive Inference Demo](#interactive-inference-demo)
- [Repository Structure](#repository-structure)
- [Installation & Getting Started](#installation--getting-started)
- [Udacity Rubric Verification](#udacity-rubric-verification)
- [Key Takeaways](#key-takeaways)
- [License & Acknowledgments](#license--acknowledgments)

---

## Project Overview

Imagine working as a Machine Learning Engineer at **CineScope**, a content discovery and streaming platform. To personalize viewer recommendations, CineScope must evaluate audience sentiment from user reviews at scale.

This project tackles that challenge by adapting a generative transformer architecture into a sequence classifier:
- Migrating from character-level encoding to **subword tokenization** via `bert-base-uncased`.
- Replacing next-token generative prediction with **global sequence mean pooling** across temporal embeddings.
- Adding a linear classification head mapping pooled representations into binary sentiment logits (`Negative` vs. `Positive`).
- Training with PyTorch `DataLoader` shuffling, `CrossEntropyLoss`, and `AdamW` optimization.
- Achieving **75.70% Test Accuracy** across 25,000 unseen test reviews, surpassing Udacity's $>75.0\%$ threshold.

---

## Key Features & Architecture

- **Subword Tokenization**: Employs WordPiece subword tokens (`bert-base-uncased`), eliminating vocabulary bottlenecks and handling unknown words via root-subword representations.
- **Custom PyTorch `IMDBDataset` & `DataLoader`**: Efficient batching, dynamic truncation, and padding to uniform length (`MAX_LENGTH = 128`).
- **Transformer Encoder (`DemoGPT`)**:
  - Embedding layer: Token embedding + Learnable Positional Embedding ($d_{\text{embed}} = 128$).
  - 4 Transformer Blocks: Multi-Head Attention (4 heads, head dimension 32) + Feed-Forward network with GELU activation and Layer Normalization.
  - Mean Pooling: Condenses sequence representations $\mathbf{X} \in \mathbb{R}^{B \times T \times d_{\text{embed}}}$ into pooled vectors $\mathbf{z} \in \mathbb{R}^{B \times d_{\text{embed}}}$.
  - Classification Head: Linear mapping from 128 hidden features to 2 class logits.
- **Stand-Out Features**:
  - Model checkpoint saving: `sentiment_model.pt`.
  - Production-ready batch inference interface: `predict_sentiment()`.

---

## Dataset & Exploratory Analysis

The dataset consists of **50,000 labeled movie reviews** from the Stanford IMDB benchmark:
- **Training Set**: 25,000 reviews (split 90/10 into 22,500 training and 2,500 validation reviews).
- **Test Set**: 25,000 reviews (12,500 positive, 12,500 negative).

### 1. Sentiment Class Distribution
Both the training and test splits exhibit an exact 50/50 balance between positive (label 1) and negative (label 0) reviews:

| Dataset Split | Negative (0) | Positive (1) | Total Samples | Class Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 12,500 | 12,500 | 25,000 | 50% / 50% |
| **Validation Set (10% of Train)** | 1,250 | 1,250 | 2,500 | 50% / 50% |
| **Test Set** | 12,500 | 12,500 | 25,000 | 50% / 50% |

### 2. Review Length & Word Count Statistics
Review lengths follow a right-skewed distribution, with a median word count of **173 words** and a median character length of **970 characters**:

```
=== Descriptive Statistics (Character & Word Lengths) ===
        char_length     word_count
count  25000.000000   25000.000000
mean    1309.431080     231.145920
std      989.722488     169.317578
min       32.000000       4.000000
25%      703.000000     125.000000
50%      970.000000     173.000000
75%     1594.000000     278.000000
max    13704.000000    2470.000000
```

*(All exploratory visualizations are rendered and embedded directly within `SentimentScope.ipynb`)*.

---

## Model Architecture Details

```
Input Tokens (Token IDs, Shape: B x 128)
  │
  ├── Token Embedding (Vocab: 30,522 -> d_embed: 128)
  └── Positional Embedding (Positions 0..127 -> d_embed: 128)
  │
  ▼
Summed Embeddings (Shape: B x 128 x 128)
  │
  ▼
[Transformer Block 1]  ── Multi-Head Self-Attention (4 heads, size 32) + FeedForward (GELU) + LayerNorm
  │
  ▼
[Transformer Block 2]  ── Multi-Head Self-Attention (4 heads, size 32) + FeedForward (GELU) + LayerNorm
  │
  ▼
[Transformer Block 3]  ── Multi-Head Self-Attention (4 heads, size 32) + FeedForward (GELU) + LayerNorm
  │
  ▼
[Transformer Block 4]  ── Multi-Head Self-Attention (4 heads, size 32) + FeedForward (GELU) + LayerNorm
  │
  ▼
Final Layer Normalization (Shape: B x 128 x 128)
  │
  ▼
Mean Pooling across Temporal Dimension (dim=1) -> (Shape: B x 128)
  │
  ▼
Linear Classification Head (128 -> 2)
  │
  ▼
Output Logits [Negative, Positive] (Shape: B x 2)
```

### Hyperparameter Configuration:
```python
config = {
    "vocabulary_size": 30522,   # bert-base-uncased vocabulary size
    "num_classes": 2,           # Binary sentiment (Negative / Positive)
    "d_embed": 128,             # Hidden embedding dimension
    "context_size": 128,        # Sequence token context window
    "layers_num": 4,            # Stacked transformer encoder blocks
    "heads_num": 4,             # Attention heads
    "head_size": 32,            # 4 heads * 32 = 128 (matches d_embed)
    "dropout_rate": 0.1,        # Dropout regularization
    "use_bias": True
}
```

---

## Training & Validation Dynamics

The model was trained for **3 epochs** on an NVIDIA GPU using the AdamW optimizer ($lr = 3 \times 10^{-4}$) and Cross-Entropy Loss:

| Epoch | Step Progress | Running Training Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: |
| **Untrained Baseline** | — | — | **48.32%** |
| **Epoch 1** | Step 100/704 $\to$ Step 700/704 | $0.6952 \to 0.5631$ | **72.28%** |
| **Epoch 2** | Step 100/704 $\to$ Step 700/704 | $0.5326 \to 0.4645$ | **77.12%** |
| **Epoch 3** | Step 100/704 $\to$ Step 700/704 | $0.4224 \to 0.4262$ | **77.52%** |

Validation accuracy increased steadily by $+29.20\%$ from the random guessing baseline, demonstrating stable convergence without overfitting.

---

## Experimental Results

The trained transformer model was evaluated on the **25,000 unseen test set reviews**:

$$\text{Final Test Accuracy} = \mathbf{75.70\%}$$

$$\text{Udacity Rubric Benchmark} > 75.0\% \quad \Longrightarrow \quad \textbf{PASSED (Exceeded)}$$

---

## Interactive Inference Demo

A simple inference interface `predict_sentiment()` allows real-time batch sentiment analysis for raw text reviews:

```python
sample_test_reviews = [
    "An absolute masterpiece! The cinematography was breathtaking and the acting was top tier.",
    "Terrible movie. The plot was completely incoherent and a total waste of two hours.",
    "A delightfully entertaining film with charming performances from the entire cast.",
    "I was utterly disappointed. Poor pacing, shallow characters, and a very predictable ending."
]

inference_results = predict_sentiment(sample_test_reviews, model, tokenizer, device)
```

### Prediction Output:
```
Review: An absolute masterpiece! The cinematography was breathtaking and the acting was top tier.
-> Predicted Sentiment: Positive (Confidence: 74.16%)
-> Probs: Neg=0.2584, Pos=0.7416

Review: Terrible movie. The plot was completely incoherent and a total waste of two hours.
-> Predicted Sentiment: Negative (Confidence: 99.99%)
-> Probs: Neg=0.9999, Pos=0.0001

Review: A delightfully entertaining film with charming performances from the entire cast.
-> Predicted Sentiment: Positive (Confidence: 94.61%)
-> Probs: Neg=0.0539, Pos=0.9461

Review: I was utterly disappointed. Poor pacing, shallow characters, and a very predictable ending.
-> Predicted Sentiment: Negative (Confidence: 99.93%)
-> Probs: Neg=0.9993, Pos=0.0007
```

---

## Repository Structure

```
AWS-Udacity-Transformers-for-Movie-Review-Sentiment-Analysis/
├── .gitignore             # Git ignore file for environments & data archives
├── CODEOWNERS             # Repository code ownership
├── LICENSE.txt            # Project MIT license
├── README.md              # Complete project documentation & reports
├── requirements.txt       # Minimal environment dependencies
├── SentimentScope.ipynb   # Complete, executed Jupyter Notebook with all outputs
└── aclImdb_v1.tar.gz      # IMDB dataset compressed archive (ignored in Git)
```

---

## Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/AWS-Udacity-Transformers-for-Movie-Review-Sentiment-Analysis.git
cd AWS-Udacity-Transformers-for-Movie-Review-Sentiment-Analysis
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook SentimentScope.ipynb
```

---

## Udacity Rubric Verification

| Section | Rubric Criterion | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Data Ingestion** | Helper function used to load train and test data | **PASS** | `load_dataset()` loads all `.txt` files with UTF-8 encoding |
| **Data Ingestion** | DataFrames have correct dimensions | **PASS** | `train_df.shape == (25000, 2)` & `test_df.shape == (25000, 2)` passed asserts |
| **EDA & Stats** | Descriptive statistics calculated | **PASS** | `info()`, `describe()`, and class counts executed |
| **EDA & Visuals** | At least 2 key metrics visualized with proper labels | **PASS** | Labeled bar charts (classes) & histogram with KDE (word lengths) |
| **PyTorch Dataset** | `__init__`, `__len__`, `__getitem__` implemented | **PASS** | `IMDBDataset` correctly returns tensor IDs `(128,)` and int labels |
| **PyTorch Dataset** | Instantiated datasets match lengths & data types | **PASS** | Length asserts (22500, 2500, 25000) and type asserts passed |
| **Model Definition** | `DemoGPT` `__init__` and `forward` adapted for classification | **PASS** | Classification head + temporal mean pooling implemented |
| **Model Definition** | Output shape matches dummy input assertions | **PASS** | Output shape `(32, 2)` matches assertion |
| **Accuracy Function** | `calculate_accuracy()` completed | **PASS** | Implemented with `torch.no_grad()`, `eval()`, returning percentage |
| **Training Loop** | Training loop completed and executes without error | **PASS** | Zero-grad, forward, loss, backward, step optimizer, loss logged |
| **Evaluation** | Test accuracy exceeds 75% | **PASS** | **75.70%** test accuracy achieved on 25,000 samples |
| **Project Report** | Results summarized with at least 2 key takeaways | **PASS** | Comprehensive summary and 3 detailed takeaways included |
| **Stand Out** | Save checkpoint & inference interface | **PASS** | `sentiment_model.pt` saved; `predict_sentiment()` demo included |

---

## Key Takeaways

1. **Mean Pooling vs. Final-Token Autoregressive Heads**: In classification tasks, aggregating representations across all sequence tokens via mean pooling provides rich contextual representations and mitigates vanishing gradient issues compared to relying solely on the final token.
2. **Subword Tokenization (WordPiece)**: Hugging Face's `bert-base-uncased` tokenizer effectively resolves the trade-off between vocabulary size and sequence length by decomposing rare words into common morphological subword units.
3. **Compact Transformers are Highly Expressive**: Even with a compact configuration ($d_{\text{embed}}=128$, 4 heads, 4 blocks), the self-attention mechanism captures nuanced semantic sentiments far more effectively than traditional n-gram or recurrent models.

---

## License & Acknowledgments

- **License**: [MIT License](LICENSE.txt).
- **Curriculum**: Designed by the [Udacity AI Programming with Python](https://www.udacity.com/) team.
- **Dataset**: Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. Association for Computational Linguistics.
