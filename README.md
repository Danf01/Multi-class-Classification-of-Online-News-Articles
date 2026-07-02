# Multi-class Classification of Online News Articles

> **Course:** Data Science and Machine Learning — Winter Project (A.Y. 2025/2026)  
> **Authors:** Francesco Cellaura (`s353978`), Daniele Ferrara (`s353385`)  
> **Institution:** Politecnico di Torino

---

## Problem Overview

Multi-class text classification of ~100,000 online news articles drawn from multiple international sources. Each article must be assigned to one of 7 categories:

| Label | Category |
|-------|----------|
| 0 | International News |
| 1 | Business |
| 2 | Technology |
| 3 | Entertainment |
| 4 | Sports |
| 5 | General News |
| 6 | Health |

The dataset is split into a **development set** (~80k articles) used for training and a held-out **evaluation set** (~20k articles) for final scoring.

---

## Pipeline Overview

```
Raw CSV
  └─► Preprocessing (missing values, ADV flags, text cleaning, timestamp, source grouping)
        └─► Feature Engineering (TF-IDF on full_content, OHE on categorical features)
              └─► ColumnTransformer
                    └─► LinearSVC (C=0.1, squared_hinge, balanced)
                          └─► submission.csv
```

---

## Repository Structure

```
news-classifier/
├── src/
│   └── pipeline.py              # Full training & prediction pipeline
├── notebooks/
│   └── Project_Pipeline_Analysis.ipynb   # EDA + hyperparameter tuning
├── results/
│   └── submission.csv           # Final predictions on evaluation set
├── data/                        # ⚠️ not tracked (see .gitignore)
│   ├── development.csv
│   └── evaluation.csv
├── requirements.txt
└── README.md
```

---

## Preprocessing

### Text (`title` + `article` → `full_content`)
- HTML tags, hyperlinks, escape sequences, digits, and non-alphanumeric characters stripped via regex
- URL path tokens extracted before removal to preserve semantic signal (e.g. `rss/sports`)
- Short tokens (≤2 chars), file extensions, and recurring technical artifacts removed
- Binary flags engineered: `is_null` (missing article body) and `is_adv` (promotional/spam titles)
- Title and article concatenated into a single `full_content` field

### Timestamp
- `ts_is_missing` binary flag for invalid timestamps (`0000-00-00 00:00:00`)
- Year and month extracted separately to capture temporal/seasonal patterns

### Source (1,300 unique values → 9 macro-categories)

| Category | Examples |
|---|---|
| `Infotainment` | CNET, Wired, PCWorld |
| `Business&Finance` | Forbes, Bloomberg |
| `Sports_Newspaper` | ESPN, CNN/SI |
| `Global_Wire` | Reuters, Xinhua |
| `Global_Broadcaster` | BBC, CNN, Al-Jazeera |
| `Digital_Portal` | Yahoo, Topix |
| `Prestige_Press` | Guardian, Washington Post |
| `Health_Magazine` | WebMD, Medical |
| `General_Magazine` | Time, Newsweek |
| `Other_Unknown` | All remaining sources |

Grouping reduces cardinality from ~1,300 to 9, avoids curse of dimensionality in OHE, and provides a stable signal from sparse sources.

---

## Feature Encoding

A `ColumnTransformer` combines two branches:

| Branch | Input | Method |
|---|---|---|
| `tfidf` | `full_content` | `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.85, stop_words='english')` |
| `cat` | `yr`, `mo`, `source_category` | `OneHotEncoder(handle_unknown='ignore')` |
| `remainder` | `is_null`, `is_adv`, `ts_is_missing` | passed through as-is |

---

## Model Selection & Hyperparameter Tuning

Two linear classifiers were evaluated:

**Logistic Regression** — L2 regularization, `lbfgs` solver, OvR strategy; C swept over `{0.7, 0.8, 0.9, 1, 2, 3}`.

**LinearSVC** — margin-based, optimized for sparse high-dimensional spaces; C swept over `{0.1, 0.3, 0.5, 0.7, 1.0, 2.0}`, loss in `{hinge, squared_hinge}`.

Both use `class_weight='balanced'` to handle class imbalance.

---

## Results

| Model | Public Score (Macro F1) |
|---|---|
| Logistic Regression (C=0.9, OvR, L2) | 0.712 |
| **LinearSVC (C=0.1, squared_hinge)** ✅ | **0.716** |

**Final model:** `LinearSVC` — selected for slightly higher public score.

### Per-class performance (LinearSVC on dev test split)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 — International News | 0.77 | 0.72 | 0.74 |
| 1 — Business | 0.73 | 0.83 | 0.78 |
| 2 — Technology | 0.79 | 0.84 | 0.81 |
| 3 — Entertainment | 0.63 | 0.49 | 0.55 |
| 4 — Sports | 0.76 | 0.96 | 0.85 |
| 5 — General News | 0.58 | 0.47 | 0.52 |
| 6 — Health | 0.57 | 0.88 | 0.69 |
| **Macro avg** | **0.69** | **0.74** | **0.71** |

Entertainment (3) and General News (5) show the lowest scores due to semantic overlap with adjacent categories and class imbalance — a known limitation of bag-of-words + linear boundary approaches.

---

## Reproduction

```bash
git clone https://github.com/<your-username>/news-classifier.git
cd news-classifier
pip install -r requirements.txt

# Place development.csv and evaluation.csv in data/
python src/pipeline.py
# → outputs results/submission.csv
```

> **Note:** `data/` is gitignored. You must supply the original dataset files.

---

## Future Work

- Lemmatization / stemming / POS tagging for richer text normalization
- Class-specific sample weighting or data augmentation for Entertainment / General News
- Contextual embeddings (e.g. **BERT**, **DistilBERT**) to capture semantic nuance beyond bag-of-words
- Ensemble methods combining linear and non-linear classifiers

---

## References

1. T. D. Science, "The ultimate preprocessing pipeline for your NLP models," 2023.
2. GeeksforGeeks, "Natural language processing (NLP) pipeline," 2023.
