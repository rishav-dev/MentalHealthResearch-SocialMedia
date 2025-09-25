import re
import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Config
POSTS_CSV = "reddit_posts.csv"
COMMENTS_CSV = "reddit_comments.csv"
MAX_WORDS = 300
WIDTH, HEIGHT = 1600, 900
BACKGROUND_COLOR = "white" 


def read_posts(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    # common text columns in our export
    text_cols = [c for c in df.columns if c.lower() in {"title","selftext","text","body"}]
    if not text_cols:
        # fallback: try all object columns
        text_cols = list(df.select_dtypes(include=["object"]).columns)
    # join row-wise text
    return df[text_cols].fillna("").agg(" ".join, axis=1)

def clean_text(s: str) -> str:
    s = s.lower()
    # remove urls
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    # remove html entities & markdown/link brackets
    s = re.sub(r"&[a-z]+;|\[|\]|\(|\)", " ", s)
    # keep letters and apostrophes
    s = re.sub(r"[^a-z'\s]", " ", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_stopwords():
    sw = set(STOPWORDS)
    # reddit/general filler tokens
    sw.update({
        "amp","nbsp","rt","https","http","www","com",
        "im","ive","dont","doesnt","didnt","cant","couldnt","wouldnt","u","ur",
        "like","just","really","also","get","got","going","one","know","think",
        "people","thing","things","make","made","much","many","even","still",
        "today","day","days","week","weeks"
    })
    # keep domain terms (do NOT add: anxiety, depression, therapy, mental, health)
    return sw

def make_wordcloud(text, out_png, title):
    if not text.strip():
        print(f"[skip] No text for {title}")
        return
    wc = WordCloud(
        width=WIDTH,
        height=HEIGHT,
        background_color=BACKGROUND_COLOR,
        max_words=MAX_WORDS,
        stopwords=build_stopwords(),
        collocations=True,
    ).generate(text)

    # Single plot for this cloud 
    plt.figure(figsize=(WIDTH/100, HEIGHT/100), dpi=100)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)
    wc.to_file(out_png)
    print(f"[saved] {out_png}")

def main():
    # Load & clean posts
    posts_series = read_posts(POSTS_CSV)
    posts_text = clean_text(" ".join(posts_series.tolist())) if not posts_series.empty else ""

    # Load & clean comments
    comments_series = read_posts(COMMENTS_CSV)
    comments_text = clean_text(" ".join(comments_series.tolist())) if not comments_series.empty else ""

    # Combined
    combined_text = " ".join([t for t in [posts_text, comments_text] if t])

    # Generate clouds
    if posts_text:
        make_wordcloud(posts_text, "wc_posts.png", "Word Cloud — Reddit Posts")
    if comments_text:
        make_wordcloud(comments_text, "wc_comments.png", "Word Cloud — Reddit Comments")
    if combined_text:
        make_wordcloud(combined_text, "wc_combined.png", "Word Cloud — Posts + Comments")

if __name__ == "__main__":
    main()
