# 😊 Text Emotion Classification (NLP)

A natural language processing project that classifies short text snippets into one of **6 emotions** using classic NLP preprocessing and machine learning models.

---

## 📁 Project Structure

```
NLP_emotion_classification/
│
├── _nlp.ipynb        # Full pipeline: EDA, text cleaning, vectorization, modeling
├── train.txt          # Labeled text dataset (text ; emotion)
└── README.md
```

---

## 📊 Dataset Overview

**File:** `train.txt`
**Format:** semicolon-separated (`text;emotion`), no header
**Size:** 16,000 rows

| Column | Type | Description |
|---|---|---|
| `text` | string | Short first-person sentence describing a feeling |
| `emotion` | categorical | One of 6 emotion labels (target variable) |

**Emotion classes:** `sadness`, `anger`, `love`, `surprise`, `fear`, `joy`

Example rows:
```
i didnt feel humiliated;sadness
i feel romantic too;love
i now feel compromised and skeptical of the value of every unit of work i put in;fear
```

No missing values were found in either column.

---

## 🧹 Text Preprocessing Pipeline

1. **Label encoding** — mapped each emotion string to an integer class (0–5)
2. **Lowercasing** — normalized all text to lowercase
3. **Punctuation removal** — stripped all punctuation via `string.punctuation`
4. **Number removal** — removed digit characters
5. **Non-ASCII / emoji removal** — filtered out non-ASCII characters
6. **Stopword removal** — removed common English stopwords using NLTK's stopword list

**Example transformation:**
```
Before: "i can go from feeling so hopeless to so damned hopeful just from being
         around someone who cares and is awake"

After:  "go feeling hopeless damned hopeful around someone cares awake"
```

---

## 🔤 Feature Extraction & Modeling

Data was split **80/20** (train/test, `random_state=42`), then vectorized and modeled using three approaches:

| Approach | Vectorizer | Model | Test Accuracy |
|---|---|---|---|
| 1 | Bag of Words (`CountVectorizer`) | Multinomial Naive Bayes | **76.8%** |
| 2 | TF-IDF (`TfidfVectorizer`) | Multinomial Naive Bayes | 66.1% |
| 3 | TF-IDF (`TfidfVectorizer`) | Logistic Regression | **86.3%** ✅ Best |

**Key takeaway:** Logistic Regression on TF-IDF features significantly outperformed both Naive Bayes variants, making it the best model for this 6-class emotion classification task.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `nltk` | Stopword removal, tokenization |
| `scikit-learn` | Vectorization (`CountVectorizer`, `TfidfVectorizer`), modeling (`MultinomialNB`, `LogisticRegression`), evaluation |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AYUSHGUPTA9506/<repo-name>.git
cd NLP_emotion_classification
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

### 3. Run the notebook
```bash
jupyter notebook _nlp.ipynb
```

> **Note:** `train.txt` must be in the same directory as the notebook. On first run, NLTK will download the `punkt` and `stopwords` resources automatically.

---

## 📈 Key Insights

- **Bag of Words outperformed TF-IDF when paired with Naive Bayes** — likely because TF-IDF's down-weighting of frequent terms hurt Naive Bayes' reliance on raw word frequency counts for short, emotion-laden text
- **Logistic Regression + TF-IDF was the clear winner**, suggesting the emotion signal in this dataset is more linearly separable in TF-IDF space than Naive Bayes' independence assumption can exploit
- Simple text cleaning (lowercasing, punctuation/number/stopword removal) was sufficient preprocessing — no stemming/lemmatization was needed to reach strong accuracy
- With 6 balanced-ish emotion classes and 16K samples, classical ML models perform competitively without needing deep learning

---

## 🔮 Future Improvements

- Try stemming/lemmatization to reduce vocabulary sparsity
- Experiment with n-grams (bigrams/trigrams) in the vectorizer
- Compare against word embeddings (Word2Vec, GloVe) or transformer-based models (BERT)
- Address class imbalance if present, using class weighting or oversampling
- Hyperparameter tuning via `GridSearchCV` for both vectorizer and model parameters

---

## 👤 Author

**Ayush Gupta**
Data Science & AI | ML · Deep Learning · Generative AI · MLOps
[GitHub](https://github.com/AYUSHGUPTA9506)
