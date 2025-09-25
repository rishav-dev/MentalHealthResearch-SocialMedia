import praw
import pandas as pd
import os

from dotenv import load_dotenv


load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

subreddits = ["depression", "Anxiety", "mentalhealth"]
posts_data = []
comments_data = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)

    for post in subreddit.search("depression OR anxiety OR mental health", limit=500):

        posts_data.append([
            sub,
            post.id,
            post.title,
            post.selftext,
            post.score,
            post.num_comments
        ])


        post.comments.replace_more(limit=0)  
        for c in post.comments.list():      
            comments_data.append([
                sub,
                post.id,
                c.id,
                c.body,
                c.score
            ])


df_posts = pd.DataFrame(posts_data, columns=["subreddit","post_id","title","text","score","num_comments"])
df_comments = pd.DataFrame(comments_data, columns=["subreddit","post_id","comment_id","comment_text","comment_score"])


df_posts.to_csv("reddit_posts.csv", index=False)
df_comments.to_csv("reddit_comments.csv", index=False)

print("Скачано постов:", len(df_posts))
print("Скачано комментариев:", len(df_comments))

