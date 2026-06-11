# Neural Network Sentiment Analysis
## E-Commerce Review Classification — Discovery-to-Action (DTA) Strategy

## Project Overview
This project builds a **TensorFlow/Keras embedding-based neural network** to classify e-commerce reviews as positive or negative sentiment, then translates model confidence scores into an automated customer support routing workflow.

## Dataset
| Property | Detail |
|----------|--------|
| Source | E-Commerce Customer Reviews |
| Raw Records | 2,120 reviews |
| Post-cleaning | 1,970 (after null drop + 3-star removal) |
| Features | `review_text` (free text), `rating` (1–5 stars) |
| Target | Binary: Positive (4–5★ → 1), Negative (1–2★ → 0) |
| Class balance | ~62% Positive / 38% Negative |

## DTA Workflow

### Phase 1 — Discovery
1. **Null handling** — 30 rows with missing `review_text` dropped
2. **3-star removal** — 120 neutral reviews dropped to sharpen classification boundary
3. **Binary labelling** — 4–5★ → 1 (Positive), 1–2★ → 0 (Negative)
4. **Text standardization plan** — lowercase + punctuation strip, vocab=10,000, max_len=100

### Phase 2 — Technical
**Vectorization:**
- `TextVectorization` adapted on training data only (leakage prevention)
- Pre-vectorized to integer arrays before training

**Architecture:**
```
Embedding(10000 × 32) → GlobalAveragePooling1D → Dense(16, ReLU) → Dense(1, Sigmoid)
```

**Training:** Adam + Binary Cross-Entropy · 10 epochs · Batch size 32 · 15% validation split

**Performance:**
| Metric | Value |
|--------|-------|
| Training Accuracy (Epoch 10) | 100.00% |
| Validation Accuracy | 100.00% |
| Test Accuracy | 100.00% |
| ROC-AUC | 1.0000 |

> **Note on perfect accuracy:** The synthetic dataset uses template-based reviews with highly consistent vocabulary. A real-world production dataset with messy, diverse language would yield more realistic (lower) accuracy, typically 85–92% for this architecture.

### Phase 3 — Action
**Required specification test:**
- Input: `"The product arrived broken and I am very unhappy"`
- Score: **0.3184** → **NEGATIVE (0)** ✅ — meets specification (output < 0.5, approaches 0)

**Threshold recommendation: 0.2 for auto-flagging**
| Score | Routing |
|-------|---------|
| < 0.20 | Auto-route to support (SLA 2 hrs) |
| 0.20–0.40 | Flag for priority human review |
| 0.40–0.60 | Human review required |
| 0.60–0.80 | Auto-acknowledge + monitor |
| > 0.80 | Auto-publish + analytics |

## Project Structure
```
ecommerce-sentiment-nn/
├── ecommerce_sentiment_nn.ipynb  # Main notebook (all cells executed)
├── ecommerce_reviews.csv         # Dataset
├── README.md                     # This file
├── eda_overview.png              # Rating distribution + text length EDA
├── training_convergence.png      # Accuracy & loss over 10 epochs
├── model_evaluation.png          # Confusion matrix + ROC curve
├── test_review_score.png         # Required specification test output
├── confidence_analysis.png       # Score distribution + threshold sweep
└── routing_workflow.png          # Automated review routing chart
```

## Setup
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook ecommerce_sentiment_nn.ipynb
```

## Author
**Abubakar Jibrin Gunda**  
AI/Data Professional · Kano State, Nigeria
