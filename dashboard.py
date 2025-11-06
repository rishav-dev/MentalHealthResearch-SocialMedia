# pip install dash plotly pandas numpy

import os
import numpy as np
import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go

#  data paths
DATA_PATHS = {
    "posts": "reddit_posts.csv",
    "comments": "reddit_comments.csv",
    "anxiety": "reddit_mental_health_anxiety_posts.csv",
    "topic_dist": "topic_distribution_clean.csv",
    "topic_info": "topic_info_clean.csv",
    "topic_examples": "topic_examples_clean.csv",
}

#  safe read
def safe_read(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None

posts = safe_read(DATA_PATHS["posts"])
comments = safe_read(DATA_PATHS["comments"])
anx = safe_read(DATA_PATHS["anxiety"])
topic_dist = safe_read(DATA_PATHS["topic_dist"])
topic_info = safe_read(DATA_PATHS["topic_info"])
topic_examples = safe_read(DATA_PATHS["topic_examples"])

#  helpers
def find_time_col(df):
    if df is None: return None
    cands = [c for c in df.columns if any(k in c.lower() for k in ["created","time","date","utc","timestamp"])]
    for c in cands:
        try:
            pd.to_datetime(df[c], errors="raise"); return c
        except Exception:
            try:
                pd.to_datetime(df[c], errors="coerce"); return c
            except Exception:
                pass
    return None

def find_id_col(df):
    if df is None: return None
    for c in df.columns:
        if c.lower() in ["id","post_id","link_id","name","doc_id","document_id"]:
            return c
    return None

def find_topic_col(df):
    if df is None: return None
    for c in df.columns:
        if "topic" in c.lower(): return c
    return None

def find_col(df, names):
    if df is None: return None
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in lower: return lower[n]
    for c in df.columns:
        if any(n in c.lower() for n in names): return c
    return None

#  infer columns from posts
post_time = find_time_col(posts)
post_id = find_id_col(posts)
post_subreddit = find_col(posts, ["subreddit"])
post_title = find_col(posts, ["title"])
post_text = find_col(posts, ["selftext","text","body","content"])
post_score = find_col(posts, ["score","ups","upvotes"])
post_ncom = find_col(posts, ["num_comments","comments"])

#  coerce datetime if present
if posts is not None and post_time:
    posts[post_time] = pd.to_datetime(posts[post_time], errors="coerce", utc=True)

#  topic label map
topic_label_map = {}
if topic_info is not None and len(topic_info):
    df_t = topic_info.copy()
    tinfo_id = None; tinfo_label = None; tinfo_kw = None
    for c in df_t.columns:
        cl = c.lower()
        if tinfo_id is None and (("topic" in cl) or cl in ["id","topic_id","label_id"]): tinfo_id = c
        if tinfo_label is None and "label" in cl: tinfo_label = c
        if tinfo_kw is None and ("keyword" in cl or "terms" in cl): tinfo_kw = c
    if tinfo_id is None:
        df_t["Topic"] = np.arange(len(df_t)); tinfo_id = "Topic"
    if tinfo_label is None:
        if tinfo_kw is not None:
            df_t["Label"] = df_t[tinfo_kw].astype(str).str.split(",").str[:3].str.join(", ")
        else:
            df_t["Label"] = "Topic " + df_t[tinfo_id].astype(str)
        tinfo_label = "Label"
    for _, r in df_t[[tinfo_id, tinfo_label]].dropna().iterrows():
        topic_label_map[r[tinfo_id]] = str(r[tinfo_label])

#  figure builders (autosize disabled, fixed heights)
def lock(fig, h=380):
    # force a stable size so graphs don't grow
    fig.update_layout(height=h, autosize=False, margin=dict(l=40, r=20, t=60, b=60), template="plotly_white")
    return fig

def fig_posts_by_subreddit(df):
    if df is None or post_subreddit is None or post_subreddit not in df.columns: return None
    c = df[post_subreddit].value_counts().reset_index()
    c.columns = ["subreddit","posts"]
    return lock(px.bar(c, x="subreddit", y="posts", title="posts by subreddit").update_layout(xaxis_tickangle=-30), 380)

def fig_score_hist(df):
    if df is None or post_score is None or post_score not in df.columns: return None
    return lock(px.histogram(df, x=post_score, nbins=40, title="score distribution"), 360)

def fig_comments_hist(df):
    if df is None or post_ncom is None or post_ncom not in df.columns: return None
    return lock(px.histogram(df, x=post_ncom, nbins=40, title="comments distribution"), 360)

def fig_score_vs_comments(df):
    if df is None or post_score is None or post_ncom is None: return None
    if post_score not in df.columns or post_ncom not in df.columns: return None
    color = post_subreddit if (post_subreddit and post_subreddit in df.columns) else None
    # no trendline to avoid statsmodels dependency
    return lock(px.scatter(df, x=post_ncom, y=post_score, color=color,
                           labels={post_ncom:"num_comments", post_score:"score"},
                           title="engagement scatter: comments vs score"), 420)

def fig_topics_treemap():
    if topic_info is not None and len(topic_info):
        df = topic_info.copy()
        tid = next((c for c in df.columns if ("topic" in c.lower()) or (c.lower() in ["id","topic_id","label_id"])), None)
        if tid is None:
            df["Topic"] = np.arange(len(df)); tid = "Topic"
        lab = next((c for c in df.columns if "label" in c.lower()), None)
        if lab is None:
            tkw = next((c for c in df.columns if ("keyword" in c.lower()) or ("terms" in c.lower())), None)
            if tkw: df["Label"] = df[tkw].astype(str).str.split(",").str[:3].str.join(", ")
            else: df["Label"] = "Topic " + df[tid].astype(str)
            lab = "Label"
        cnt = next((c for c in df.columns if c.lower() in ["count","n","frequency","freq","size","documents"]), None)
        if cnt is None and topic_dist is not None:
            td_topic = find_topic_col(topic_dist); td_id = find_id_col(topic_dist)
            if td_topic is not None and td_id is not None:
                counts = topic_dist.groupby(td_topic).size().rename("Count").reset_index()
                df = df.merge(counts, left_on=tid, right_on=td_topic, how="left"); cnt = "Count"
        if cnt is None: df["Count"] = 1; cnt = "Count"
        df["LabelShort"] = df[lab].astype(str).str.slice(0,50)
        return lock(px.treemap(df, path=["LabelShort"], values=cnt, title="topic landscape (treemap)"), 500)
    if topic_dist is not None:
        td_topic = find_topic_col(topic_dist)
        if td_topic is not None:
            counts = topic_dist.groupby(td_topic).size().rename("Count").reset_index()
            counts["LabelShort"] = "Topic " + counts[td_topic].astype(str)
            return lock(px.treemap(counts, path=["LabelShort"], values="Count", title="topic landscape (treemap)"), 500)
    return None

def fig_topic_engagement(df, allowed_topics=None):
    if df is None or topic_dist is None: return None
    td_topic = find_topic_col(topic_dist); td_id = find_id_col(topic_dist)
    if td_topic is None or td_id is None or post_id is None: return None
    td = topic_dist.copy()
    if allowed_topics is not None:
        td = td[td[td_topic].isin(list(allowed_topics))]
    if td.empty: return None
    merged = df.merge(td[[td_id, td_topic]], left_on=post_id, right_on=td_id, how="left")
    if merged.empty: return None
    score_col = post_score if (post_score in df.columns) else None
    ncom_col = post_ncom if (post_ncom in df.columns) else None
    agg = merged.groupby(td_topic).agg(
        avg_score=(score_col,"mean") if score_col else (td_topic,"size"),
        avg_comments=(ncom_col,"mean") if ncom_col else (td_topic,"size"),
        n=(td_topic,"size"),
    ).reset_index()
    if agg.empty: return None
    agg["TopicLabel"] = agg[td_topic].map(topic_label_map).fillna("Topic " + agg[td_topic].astype(str))
    xcol = "avg_comments" if "avg_comments" in agg.columns else "n"
    ycol = "avg_score" if "avg_score" in agg.columns else "n"
    return lock(px.scatter(agg, x=xcol, y=ycol, size="n", hover_name="TopicLabel",
                           title="topic engagement (avg comments × avg score)"), 460)

def fig_top_keywords_bar():
    if topic_info is None: return None
    key_col = next((c for c in topic_info.columns if "keyword" in c.lower() or "terms" in c.lower()), None)
    if key_col is None: return None
    s = (topic_info[key_col].dropna().astype(str).str.split(",").explode().str.strip().str.lower())
    top = s.value_counts().head(20).reset_index(); top.columns = ["keyword","freq"]
    return lock(px.bar(top, x="keyword", y="freq", title="top topic keywords").update_layout(xaxis_tickangle=-45), 380)

def fig_top_posts_table(df, k=20):
    if df is None or post_score not in df.columns or post_title not in df.columns: return None
    t = df.sort_values(post_score, ascending=False).head(k).copy()
    show = {post_title:"title"}
    if post_score: show[post_score] = "score"
    if post_ncom and post_ncom in t.columns: show[post_ncom] = "num_comments"
    if post_subreddit and post_subreddit in t.columns: show[post_subreddit] = "subreddit"
    t = t[list(show.keys())].rename(columns=show)
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in t.columns],
        data=t.to_dict("records"),
        page_size=10,
        style_table={"overflowX":"auto"},
        style_cell={"backgroundColor":"#0f1117","color":"#eaeef2","border":"1px solid #222636"},
        style_header={"backgroundColor":"#121317","fontWeight":"bold"},
    )

#  app
app = Dash(__name__, title="Reddit Insights Dashboard", suppress_callback_exceptions=True)

#  index shell
app.index_string = f"""
<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>Reddit Insights Dashboard</title>
    {{%favicon%}}
    {{%css%}}
    <style>
    :root{{ --bg:#0b0c10; --card:#121317; --ink:#eaeef2; --muted:#9aa3af; --accent:#7c3aed; }}
    *{{box-sizing:border-box}}
    body{{margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;}}
    .app{{background:var(--bg); color:var(--ink); min-height:100vh;}}
    .header{{padding:28px 22px; position:sticky; top:0; background:linear-gradient(180deg,rgba(11,12,16,.9),rgba(11,12,16,.7)); backdrop-filter: blur(8px); border-bottom:1px solid #1f2430; z-index:100;}}
    .h1{{font-size:28px; font-weight:800; letter-spacing:.2px; display:flex; align-items:center; gap:10px}}
    .pill{{font-size:12px; background:linear-gradient(135deg,#a78bfa,#f472b6); padding:6px 10px; border-radius:999px; color:#fff}}
    .sub{{color:var(--muted); margin-top:6px}}
    .container{{max-width:1280px; margin:0 auto}}
    .controls{{display:grid; gap:12px; grid-template-columns:1fr 1fr 1fr; padding:12px 0 4px 0}}
    @media(min-width:980px){{ .controls{{grid-template-columns:1fr 1fr 1fr}} }}
    select, input[type=text]{{width:100%; padding:10px 12px; border-radius:10px; border:1px solid #222636; background:#0f1117; color:#eaeef2;}}
    .card{{background:linear-gradient(180deg, #161821, #0f1117); border:1px solid #222636; border-radius:18px; padding:14px; margin:12px 0}}
    .card h2{{font-size:18px; font-weight:700; margin:6px 10px 10px 10px}}
    .grid{{display:grid; gap:18px; grid-template-columns:1fr}}
    @media(min-width:980px){{ .grid{{grid-template-columns:1fr 1fr}} }}
    .footer{{color:#a0a8b7; font-size:12px; text-align:center; padding:30px 0 60px}}
    </style>
  </head>
  <body>
    <div class="app">
      {{%app_entry%}}
    </div>
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""

#  dropdown values
all_subreddits = ["__ALL__"]
if posts is not None and post_subreddit in (posts.columns if posts is not None else []):
    opts = [s for s in posts[post_subreddit].dropna().unique().tolist() if str(s).strip() != ""]
    all_subreddits = ["__ALL__"] + sorted(opts)

#  layout
app.layout = html.Div(className="app", children=[
    html.Div(className="header", children=[
        html.Div(className="container", children=[
            html.Div(className="h1", children=["Reddit Insights Dashboard", html.Span("Interactive • Plotly", className="pill")]),
            html.Div(className="sub", children="only charts supported by your data are shown. no broken cards.")
        ])
    ]),
    html.Div(className="container", children=[
        html.Div(className="controls", children=[
            html.Div([
                html.Label("subreddit", style={"color":"#cbd5e1","fontSize":"13px"}),
                dcc.Dropdown(id="subreddit",
                             options=[{"label": s if s!="__ALL__" else "All", "value": s} for s in all_subreddits],
                             value="__ALL__", clearable=False)
            ]),
            html.Div([
                html.Label("topic search (label contains)", style={"color":"#cbd5e1","fontSize":"13px"}),
                dcc.Input(id="topic-search", type="text", placeholder="e.g., anxiety, relationship", debounce=True)
            ]),
            html.Div([
                html.Label("refresh", style={"color":"#cbd5e1","fontSize":"13px"}),
                html.Button("update charts", id="refresh", n_clicks=0,
                            style={"padding":"8px 12px","borderRadius":"8px","background":"#0b0d12","border":"1px solid #222636","color":"#eaeef2","cursor":"pointer"})
            ]),
        ]),
        html.Div(id="cards", className="grid"),
        html.Div(className="footer", children="made with ❤️ plotly • hover for details, drag to zoom, export pngs from the toolbar.")
    ])
])

#  filtering
def apply_filters(posts_df, subreddit):
    if posts_df is None: return None
    df = posts_df.copy()
    if subreddit and post_subreddit and subreddit != "__ALL__":
        df = df[df[post_subreddit] == subreddit]
    return df

#  callback builds only supported charts and fixes height per graph
@app.callback(
    Output("cards", "children"),
    Input("refresh", "n_clicks"),
    State("subreddit", "value"),
    State("topic-search", "value"),
    prevent_initial_call=False
)
def build_cards(n, subreddit_value, topic_query):
    df = apply_filters(posts, subreddit_value)

    figs = []
    f1 = fig_posts_by_subreddit(df);           f1 and figs.append(("Posts by Subreddit",    f1, 380))
    f2 = fig_score_hist(df);                   f2 and figs.append(("Score Distribution",    f2, 360))
    f3 = fig_comments_hist(df);                f3 and figs.append(("Comments Distribution", f3, 360))
    f4 = fig_score_vs_comments(df);            f4 and figs.append(("Engagement Scatter (Comments vs Score)", f4, 420))
    f5 = fig_topics_treemap();                 f5 and figs.append(("Topic Landscape (Treemap)", f5, 500))

    allowed = None
    if topic_query:
        allowed = {tid for tid, lab in topic_label_map.items() if topic_query.lower() in lab.lower()}
        if not allowed: allowed = None
    f6 = fig_topic_engagement(df, allowed_topics=allowed); f6 and figs.append(("Topic Engagement", f6, 460))

    f7 = fig_top_keywords_bar();               f7 and figs.append(("Top Topic Keywords",    f7, 380))

    table = fig_top_posts_table(df, k=20)

    cards = []
    for title, content, h in figs:
        cards.append(
            html.Div(className="card", children=[
                html.H2(title),
                dcc.Graph(figure=content, style={"height": f"{h}px"}, config={"responsive": False})
            ])
        )
    if table is not None:
        cards.append(html.Div(className="card", children=[html.H2("Top Posts (by score)"), table]))

    if len(cards) < 3:
        cards.append(html.Div(className="card", children=[html.H2("note"), html.Div("not enough columns for more charts in the current csvs.")]))

    return cards

if __name__ == "__main__":
    print("walkthrough (2–3 min): subreddit mix → score/comments distributions → engagement scatter → topic treemap → topic engagement → top keywords → top posts table.")
    app.run(debug=True)
