# 🛍️ Amazon Sentiment Classification (BERT + Traditional ML)

This project applies both traditional machine learning and fine-tuned BERT to classify sentiment in Amazon product reviews. It explores preprocessing, vectorization, class imbalance, and evaluation metrics to achieve more balanced and nuanced predictions.

---

## 🧠 Project Summary

- **Goal:** Classify Amazon product reviews as Positive, Neutral, or Negative
- **Dataset:** Amazon product review dataset with fields like `reviews.text`, `reviews.rating`, and derived `sentiment`
- **Approach:**
  - Traditional ML (TF-IDF + Logistic Regression / other classifiers)
  - Fine-tuned `bert-base-uncased` model
  - Addressed class imbalance with:
    - Class weighting
    - Random oversampling
    - Balanced CrossEntropy loss

---

## 📦 Tech Stack

- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn
- BERT via Hugging Face Transformers
- PyTorch (for model fine-tuning)
- Jupyter Notebook

---

## 🗂️ Folder Structure

```
Supervised_Sentiment_Classification/
├── data/
│   └── amazon_data.csv         # (gitignored)
├── notebooks/
│   └── Sentiment_Classifier.ipynb
├── reports/
│   ├── Sentiment Classification of Amazon Data.pdf
│   └── Project Report.docx     # (gitignored)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 📊 Model Results

| Model             | Accuracy | F1-Score | Notes                           |
| ----------------- | -------- | -------- | ------------------------------- |
| Traditional ML    | \~70%    | 68–70%   | Struggled with class imbalance  |
| BERT (Unweighted) | 71%      | 48%      | Over-predicted Positive class   |
| BERT (Weighted)   | 66.8%    | 66%      | More balanced class performance |

## 🧪 How to Run
```
# Clone the repo
git clone https://github.com/yourusername/Amazon_Sentiment_Classification.git
cd Amazon_Sentiment_Classification

# Create .env file (no keys needed for this project)
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/Sentiment_Classifier.ipynb
```

## 🔐 Environment Variables
- No API keys are required for this project.


