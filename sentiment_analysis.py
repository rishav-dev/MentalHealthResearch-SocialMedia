"""
Sentiment analysis on Reddit mental health data.
"""

# Imports and setup
#pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob transformers torch
#python -m textblob.download_corpora


import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
RANDOM_STATE = 42

# Make sure NLTK stuff is available
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Data loading

DATA_PATHS = {
    "posts": "reddit_posts.csv",
    "comments": "reddit_comments.csv",
    "anxiety": "reddit_mental_health_anxiety_posts.csv",
}

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return None

posts_df = safe_read_csv(DATA_PATHS["posts"])
comments_df = safe_read_csv(DATA_PATHS["comments"])
anx_df = safe_read_csv(DATA_PATHS["anxiety"])  # main for trends (has timestamp)

# Text preprocessing
def basic_clean(text: str) -> str:
    """Lowercase, remove URLs, non-alphabetic chars, and extra spaces."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z\s']", " ", text)                 # keep letters and '
    text = re.sub(r"\s+", " ", text).strip()               # normalize spaces
    return text

# Apply cleaning to relevant text columns
if posts_df is not None:
    posts_df["clean_text"] = posts_df["text"].astype(str).apply(basic_clean)

if comments_df is not None:
    comments_df["clean_text"] = comments_df["comment_text"].astype(str).apply(basic_clean)

if anx_df is not None:
    # selftext + title combined for richer signal
    anx_df["full_text"] = (
        anx_df["title"].fillna("") + " " +
        anx_df["selftext"].fillna("")
    )
    anx_df["clean_text"] = anx_df["full_text"].astype(str).apply(basic_clean)

# Lexicon & Transformer Sentiment

# TextBlob polarity

def textblob_polarity(text: str) -> float:
    """Return polarity in [-1, 1] from TextBlob."""
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity

# VADER compound score

sia = SentimentIntensityAnalyzer()

def vader_compound(text: str) -> float:
    """Return VADER compound score in [-1, 1]."""
    if not text:
        return 0.0
    return sia.polarity_scores(text)["compound"]

# Transformer's sentiment: Using distilbert-base-uncased-finetuned-sst-2-english by default

print("[INFO] Initializing HuggingFace sentiment pipeline (this may take a bit on first run)...")
hf_classifier = pipeline("sentiment-analysis")

def hf_sentiment_label_and_score(texts, batch_size=16, max_length=512):
    """
    Run HF sentiment classification in batches.
    - Truncates long texts so they fit the model's max length.
    Returns two lists: labels, scores.
    """
    all_labels = []
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch_raw = texts[i:i+batch_size]
        batch = []
        for t in batch_raw:
            t = str(t)
            # keep only first 300 words to be safe
            words = t.split()
            if len(words) > 300:
                t = " ".join(words[:300])
            batch.append(t)

        # Tell the pipeline to truncate to the model's max_length
        results = hf_classifier(
            batch,
            truncation=True, 
            max_length=max_length
        )

        for r in results:
            all_labels.append(r["label"])   # "POSITIVE" or "NEGATIVE"
            all_scores.append(r["score"])   # confidence

    return all_labels, all_scores



# Apply sentiment models to datasets
def apply_sentiment_models(df: pd.DataFrame, text_col: str, prefix: str):
    """
    Adds columns:
    - {prefix}_tb_polarity
    - {prefix}_vader_compound
    - {prefix}_hf_label
    - {prefix}_hf_score
    """
    if df is None or text_col not in df.columns:
        print(f"[WARN] Skipping {prefix}: {text_col} not in df")
        return df

    print(f"[INFO] Applying sentiment models to {prefix} on column '{text_col}'")

    texts = df[text_col].astype(str).tolist()

    # TextBlob
    df[f"{prefix}_tb_polarity"] = df[text_col].astype(str).apply(textblob_polarity)

    # VADER
    df[f"{prefix}_vader_compound"] = df[text_col].astype(str).apply(vader_compound)

    # HF / BERT-like
    labels, scores = hf_sentiment_label_and_score(texts)
    df[f"{prefix}_hf_label"] = labels
    df[f"{prefix}_hf_score"] = scores

    return df

# Applying to each dataset
posts_df = apply_sentiment_models(posts_df, "clean_text", "post")
comments_df = apply_sentiment_models(comments_df, "clean_text", "comment")
anx_df = apply_sentiment_models(anx_df, "clean_text", "anx")

# Mapping continuous scores to discrete labels

def score_to_label(score: float, pos_thresh=0.05, neg_thresh=-0.05) -> str:
    """
    Map polarity/compound score to discrete label.
    """
    if score >= pos_thresh:
        return "positive"
    elif score <= neg_thresh:
        return "negative"
    else:
        return "neutral"

def add_discrete_labels(df, prefix: str):
    if df is None:
        return df
    tb_col = f"{prefix}_tb_polarity"
    vd_col = f"{prefix}_vader_compound"
    if tb_col in df.columns:
        df[f"{prefix}_tb_label"] = df[tb_col].apply(score_to_label)
    if vd_col in df.columns:
        df[f"{prefix}_vader_label"] = df[vd_col].apply(score_to_label)
    return df

posts_df = add_discrete_labels(posts_df, "post")
comments_df = add_discrete_labels(comments_df, "comment")
anx_df = add_discrete_labels(anx_df, "anx")

# HF labels consistency:
def normalize_hf_label(label: str) -> str:
    label = str(label).upper()
    if "POS" in label:
        return "positive"
    elif "NEG" in label:
        return "negative"
    return "neutral"

for df, prefix in [(posts_df, "post"), (comments_df, "comment"), (anx_df, "anx")]:
    if df is not None and f"{prefix}_hf_label" in df.columns:
        df[f"{prefix}_hf_label_norm"] = df[f"{prefix}_hf_label"].apply(normalize_hf_label)


# experiment: how well do lexicon models match bert?
def evaluate_against_hf(df: pd.DataFrame, prefix: str, model_col: str):
    """
    Compare a lexicon model's labels to normalized HF labels.
    model_col: e.g., f"{prefix}_tb_label" or f"{prefix}_vader_label"
    """
    hf_col = f"{prefix}_hf_label_norm"
    if df is None or hf_col not in df.columns or model_col not in df.columns:
        print(f"[WARN] Could not evaluate {model_col} in {prefix}")
        return None

    sub = df[[hf_col, model_col]].dropna()
    y_true = sub[hf_col]
    y_pred = sub[model_col]

    print(f"\n=== Agreement: {model_col} vs HF ({prefix}) ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))

    return {
        "model": model_col,
        "accuracy": accuracy_score(y_true, y_pred)
    }

lexicon_eval_results = []

lexicon_eval_results.append(
    evaluate_against_hf(posts_df, "post", "post_tb_label")
)
lexicon_eval_results.append(
    evaluate_against_hf(posts_df, "post", "post_vader_label")
)

lexicon_eval_results.append(
    evaluate_against_hf(anx_df, "anx", "anx_tb_label")
)
lexicon_eval_results.append(
    evaluate_against_hf(anx_df, "anx", "anx_vader_label")
)

# ML models trained on text to predict HF sentiment
"""
We will:
- Use the anxiety dataset (has timestamps) as our main supervised set
- Use HF normalized labels ("positive"/"negative") as pseudo ground truth
- Train classical models on TF-IDF features:
    - Logistic Regression
    - Linear SVM
    - Random Forest
- Evaluate Accuracy, Precision, Recall, F1
"""

ml_results = []

def train_and_evaluate_ml_models(df: pd.DataFrame, text_col: str, label_col: str):
    global ml_results
    if df is None:
        print("[WARN] No dataframe for ML models")
        return

    # Keep only rows with non-empty text and labels in {positive, negative}
    data = df[[text_col, label_col]].dropna()
    data = data[data[text_col].str.strip() != ""]
    data = data[data[label_col].isin(["positive", "negative"])]

    print(f"\n[INFO] ML training dataset shape: {data.shape}")
    print(data[label_col].value_counts())

    X_text = data[text_col].astype(str).tolist()
    y = data[label_col].tolist()

    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVM": LinearSVC(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        )
    }

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} Precision: {precision:.4f}")
        print(f"{name} Recall:    {recall:.4f}")
        print(f"{name} F1-score:  {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=3))

        ml_results.append({
            "model": name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # Return tfidf + last model for optional later use
    return tfidf, models

# Train ML models on anxiety dataset using HF sentiment as labels
if anx_df is not None:
    tfidf_vectorizer, trained_models = train_and_evaluate_ml_models(
        anx_df, text_col="clean_text", label_col="anx_hf_label_norm"
    )

# Visualizations


OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sentiment distribution (HF, anxiety dataset)

if anx_df is not None and "anx_hf_label_norm" in anx_df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="anx_hf_label_norm", data=anx_df, order=["negative", "neutral", "positive"])
    plt.title("HF Sentiment Distribution (Anxiety Dataset)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anxiety_hf_sentiment_distribution.png"), dpi=150)
    plt.close()
    print(f"[INFO] Saved: {OUTPUT_DIR}/anxiety_hf_sentiment_distribution.png")

# Sentiment trend over time (using HF scores on anxiety dataset)

if anx_df is not None and "created_iso" in anx_df.columns:
    tmp = anx_df.copy()
    tmp["created_dt"] = pd.to_datetime(tmp["created_iso"], errors="coerce")
    tmp = tmp.dropna(subset=["created_dt"])

    # Converting HF label to +1 / -1; neutral as 0
    def label_to_signed(label):
        label = normalize_hf_label(label)
        if label == "positive":
            return 1
        elif label == "negative":
            return -1
        else:
            return 0

    tmp["hf_sentiment_signed"] = tmp["anx_hf_label"].apply(label_to_signed)

    # Aggregate by day
    tmp["date"] = tmp["created_dt"].dt.date
    daily = tmp.groupby("date")["hf_sentiment_signed"].mean().reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(daily["date"], daily["hf_sentiment_signed"], marker="o", linewidth=1)
    plt.xticks(rotation=45)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Average HF Sentiment Over Time (Anxiety Dataset)")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment (−1 = neg, +1 = pos)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anxiety_sentiment_trend_over_time.png"), dpi=150)
    plt.close()
    print(f"[INFO] Saved: {OUTPUT_DIR}/anxiety_sentiment_trend_over_time.png")

# Subreddit sentiment comparison (posts)

if posts_df is not None and "post_hf_label_norm" in posts_df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(
        x="subreddit",
        hue="post_hf_label_norm",
        data=posts_df,
        order=sorted(posts_df["subreddit"].dropna().unique())
    )
    plt.title("HF Sentiment by Subreddit (Posts)")
    plt.xlabel("Subreddit")
    plt.ylabel("Count")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "posts_sentiment_by_subreddit.png"), dpi=150)
    plt.close()
    print(f"[INFO] Saved: {OUTPUT_DIR}/posts_sentiment_by_subreddit.png")

# ML model performance comparison bar chart

if len(ml_results) > 0:
    ml_df = pd.DataFrame(ml_results)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="model", y="accuracy", data=ml_df)
    plt.title("ML Model Accuracy (Predicting HF Sentiment)")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ml_model_accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"[INFO] Saved: {OUTPUT_DIR}/ml_model_accuracy_comparison.png")

    plt.figure(figsize=(8, 5))
    sns.barplot(x="model", y="f1", data=ml_df)
    plt.title("ML Model F1-score (Predicting HF Sentiment)")
    plt.ylabel("F1-score")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ml_model_f1_comparison.png"), dpi=150)
    plt.close()
    print(f"[INFO] Saved: {OUTPUT_DIR}/ml_model_f1_comparison.png")

# Save enriched data & metrics

if posts_df is not None:
    posts_df.to_csv("reddit_posts_with_sentiment.csv", index=False)
    print("[INFO] Saved reddit_posts_with_sentiment.csv")

if comments_df is not None:
    comments_df.to_csv("reddit_comments_with_sentiment.csv", index=False)
    print("[INFO] Saved reddit_comments_with_sentiment.csv")

if anx_df is not None:
    anx_df.to_csv("reddit_mental_health_anxiety_posts_with_sentiment.csv", index=False)
    print("[INFO] Saved reddit_mental_health_anxiety_posts_with_sentiment.csv")

if len(ml_results) > 0:
    pd.DataFrame(ml_results).to_csv("ml_model_results.csv", index=False)
    print("[INFO] Saved ml_model_results.csv")

print("\n[DONE] Sentiment analysis pipeline complete.")
