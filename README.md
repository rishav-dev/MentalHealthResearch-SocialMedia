# Mental Health Discussions on Reddit

Analysis of 25,886 Reddit posts and comments from r/Anxiety, r/depression, and r/mentalhealth using sentiment analysis, topic modeling, and supervised classification.

The project compares several ways of labeling and modeling the text, and includes a Plotly Dash interface for exploring the results.

## What I found

**The conversation drifts well outside mental-health vocabulary.** Topic modeling on 7,469 cleaned items produced two broad clusters about mental health, therapy, and medication. The next-largest themes were not clinical: news and politics (228 items), social media (201), money (48), and housing, including apartments and landlords (30). People go to these subreddits to talk about anxiety, and a lot of what they talk about is the pressure around it: rent, money, the news, and their feeds.

**Classifiers agree because the labels are easy to match.** Three classifiers trained on TF-IDF features landed within a point of each other. They predict labels produced by a pretrained sentiment model, so high accuracy mostly shows they can reproduce that model, not that they understand the posts.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9125 | 0.8327 | 0.9125 | 0.8708 |
| Random Forest | 0.9125 | 0.8327 | 0.9125 | 0.8708 |
| Linear SVM | 0.9063 | 0.8322 | 0.9063 | 0.8676 |

Precision, recall, and F1 are weighted averages. Full results are in `ml_model_results.csv`.

## Data

6,398 posts and 19,488 comments collected with PRAW from r/Anxiety, r/depression, and r/mentalhealth, plus an additional set of anxiety posts. Raw and scored CSVs are both in the repository.

## Method

1. **Collection** (`reddit_data_collection.py`): PRAW search across the three subreddits. Credentials are read from a `.env` file.
2. **Sentiment** (`sentiment_analysis.py`): every item scored three ways: TextBlob polarity, VADER compound score, and a Hugging Face sentiment pipeline. The lexicon labels are compared against the transformer labels.
3. **Classification** (`sentiment_analysis.py`): logistic regression, linear SVM, and random forest on TF-IDF (unigrams and bigrams, 5,000 features) predicting the transformer label, 80/20 stratified split.
4. **Topic modeling** (`topic_modeling.py`): BERTopic with `all-MiniLM-L6-v2` embeddings and KeyBERT-inspired topic labels, reduced to 10 topics after cleaning and language filtering.
5. **Dashboard** (`dashboard.py`): Plotly Dash app over the scored data and topic outputs.

`word_cloud.py` and the `wc_*.png` files are word clouds for posts, comments, and both combined. `figures/` holds the sentiment and model-comparison charts. `DataVisualization-Group4.mp4` is a walkthrough video.

## Run it

```bash
pip install pandas numpy scikit-learn nltk textblob transformers torch bertopic langdetect emoji matplotlib seaborn dash plotly praw python-dotenv

python sentiment_analysis.py
python topic_modeling.py
python dashboard.py
```

Re-collecting data needs Reddit API credentials in a `.env` file (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`). The collected CSVs are included, so you can skip `reddit_data_collection.py`.
