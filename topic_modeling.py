import re, emoji
import pandas as pd
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import matplotlib.pyplot as plt
from collections import Counter

DetectorFactory.seed = 0


df = pd.read_csv("reddit_comments.csv")
texts_raw = df["comment_text"].astype(str).fillna("")


url_re = re.compile(r"http\S+|www\.\S+")
multi_space = re.compile(r"\s+")
punct_re = re.compile(r"[^a-zA-Z\s]")

def clean_text(s: str) -> str:
    s = s.replace("[deleted]", "").replace("[removed]", "")
    s = url_re.sub(" ", s)
    s = emoji.replace_emoji(s, replace=" ")
    s = s.lower()
    s = punct_re.sub(" ", s)  
    s = multi_space.sub(" ", s).strip()
    return s

texts = [clean_text(t) for t in texts_raw]


def is_english(s: str) -> bool:
    if len(s) < 15:
        return True
    try:
        return detect(s) == "en"
    except:
        return False

banals = {"thank you", "thanks", "thank u", "ty", "ok", "okay"}
texts = [t for t in texts if len(t.split()) >= 8 and t not in banals and is_english(t)]

print("Осталось комментариев после фильтров:", len(texts))


extra_stops = {
    "you","your","yours","im","ive","id","dont","didnt","cant","couldnt","wouldnt","ill",
    "like","really","just","thing","things","something","someone","anyone","everyone",
    "got","get","gets","gotten","going","went","u","ur","amp"
}
stopwords = ENGLISH_STOP_WORDS.union(extra_stops)

vectorizer_model = CountVectorizer(
    stop_words=list(stopwords),
    ngram_range=(1, 2),
    min_df=2,     
    max_df=0.9    
)

representation_model = KeyBERTInspired()


topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    language="english",
    nr_topics=None,                
    calculate_probabilities=True,
    verbose=True
)


topics, probs = topic_model.fit_transform(texts)


topic_model.reduce_topics(texts, nr_topics=10)


topic_info = topic_model.get_topic_info()
topic_info.to_csv("topic_info_clean.csv", index=False)
print(topic_info.head(10))


valid_topics = [t for t in topics if t != -1]
counts = Counter(valid_topics)
dist = pd.DataFrame(
    sorted(counts.items(), key=lambda x: x[1], reverse=True),
    columns=["Topic","Count"]
)
name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
dist["Name"] = dist["Topic"].map(name_map)
dist.to_csv("topic_distribution_clean.csv", index=False)

plt.figure(figsize=(10,5))
plt.bar(dist["Name"].astype(str), dist["Count"])
plt.xticks(rotation=45, ha="right")
plt.title("Topic distribution (cleaned, BERTopic)")
plt.tight_layout()
plt.savefig("topic_distribution_clean.png", dpi=200)
plt.show()


rows = []
for t in dist["Topic"].head(5):
    reps = topic_model.get_representative_docs(t)
    for i, txt in enumerate(reps[:5], 1):
        rows.append({"Topic": t, "Name": name_map.get(t), "Rank": i, "Comment": txt})
pd.DataFrame(rows).to_csv("topic_examples_clean.csv", index=False)
