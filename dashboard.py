# pip install dash plotly pandas numpy

import os
import numpy as np
import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go

# data paths
DATA_PATHS = {
    "posts": "reddit_posts.csv",
    "comments": "reddit_comments.csv",
    "anxiety": "reddit_mental_health_anxiety_posts.csv",
    "topic_dist": "topic_distribution_clean.csv",
    "topic_info": "topic_info_clean.csv",
    "topic_examples": "topic_examples_clean.csv",
}

# safe read
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
topic_dist = topic_dist[topic_dist["Name"].notna()]
topic_info = safe_read(DATA_PATHS["topic_info"])
topic_examples = safe_read(DATA_PATHS["topic_examples"])

# helpers
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

# infer post columns
post_time = find_time_col(posts)
post_id = find_id_col(posts)
post_subreddit = find_col(posts, ["subreddit"])
post_title = find_col(posts, ["title"])
post_text = find_col(posts, ["selftext","text","body","content"])
post_score = find_col(posts, ["score","ups","upvotes"])
post_ncom = find_col(posts, ["num_comments","comments"])

# coerce datetime if present
if posts is not None and post_time:
    posts[post_time] = pd.to_datetime(posts[post_time], errors="coerce", utc=True)

# topic dist keys
td_topic = find_topic_col(topic_dist) if topic_dist is not None else None
td_id = find_id_col(topic_dist) if topic_dist is not None else None

# topic label map
topic_label_map = {}
if topic_info is not None and len(topic_info):
    df_t = topic_info.copy()
    tinfo_id = None; tinfo_label = None; tinfo_kw = None
    for c in df_t.columns:
        cl = c.lower()
        if tinfo_id is None and (("topic" in cl) or cl in ["id","topic_id","label_id"]): tinfo_id = c
        if tinfo_label is None and "Name" in cl: tinfo_label = c
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
        topic_label_map[str(r[tinfo_id])] = str(r[tinfo_label])

# topic dropdown options
def get_topic_options():
    opts = []
    if topic_info is not None and len(topic_info):
        df = topic_info.copy()
        tid = next((c for c in df.columns if ("topic" in c.lower()) or (c.lower() in ["id","topic_id","label_id"])), None)
        lab = next((c for c in df.columns if "name" in c.lower()), None)
        if tid is None:
            df["Topic"] = np.arange(len(df)); tid = "Topic"
        if lab is None:
            tkw = next((c for c in df.columns if ("keyword" in c.lower()) or ("terms" in c.lower())), None)
            if tkw: df["Label"] = df[tkw].astype(str).str.split(",").str[:3].str.join(", ")
            else: df["Label"] = "Topic " + df[tid].astype(str)
            lab = "Label"
        for _, r in df[[tid, lab]].dropna().iterrows():
            opts.append({"label": str(r[lab])[:60], "value": str(r[tid])})
    elif topic_dist is not None and td_topic is not None:
        vals = pd.Series(topic_dist[td_topic].astype(str).unique())
        for v in vals.sort_values():
            opts.append({"label": f"Topic {v}", "value": str(v)})
    return opts

TOPIC_OPTIONS = get_topic_options()

# layout helpers
def lock(fig, h=380):
    # keep strict height and make width follow the card
    fig.update_layout(
        height=h,
        autosize=False,
        margin=dict(l=40, r=20, t=60, b=60),
        template="plotly_white",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#eaeef2"),
    )
    return fig

# filtering
def filter_posts(posts_df, subreddit, selected_topic_values):
    df = posts_df.copy() if posts_df is not None else None
    if df is None: return None
    if subreddit and post_subreddit and subreddit != "__ALL__":
        df = df[df[post_subreddit] == subreddit]
    """ Unclear what is trying to be done here
    if selected_topic_values and (td_id and post_id and td_topic and topic_dist is not None):
        want = set(map(str, selected_topic_values))
        td = topic_dist.copy()
        td[td_topic] = td[td_topic].astype(str)
        td_sub = td[td[td_topic].isin(want)][[td_id, td_topic]].dropna()
        df = df.merge(td_sub, left_on=post_id, right_on=td_id, how="inner")
    """
    return df

def filter_topic_dist(selected_topic_values):
    if topic_dist is None or td_topic is None: return None
    if not selected_topic_values: return topic_dist.copy()
    want = set(map(str, selected_topic_values))
    td = topic_dist.copy()
    td[td_topic] = td[td_topic].astype(str)
    out  = td[td[td_topic].isin(want)]
    print(f"Выбрано {len(out)} строк из {len(topic_dist)} (фильтр по {want})")
    return td[td[td_topic].isin(want)]

def filter_topic_info(selected_topic_values):
    if topic_info is None or td_topic is None: return None
    if not selected_topic_values: return topic_info.copy()
    want = set(map(str, selected_topic_values))
    td = topic_info.copy()
    td[td_topic] = td[td_topic].astype(str)
    out  = td[td[td_topic].isin(want)]
    print(f"Выбрано {len(out)} строк из {len(topic_info)} (фильтр по {want})")
    return td[td[td_topic].isin(want)]


# figures
def fig_posts_by_subreddit(df):
    if df is None or post_subreddit is None or post_subreddit not in df.columns: return None
    c = df[post_subreddit].value_counts().reset_index()
    c.columns = ["subreddit","posts"]
    return lock(px.bar(c, x="subreddit", y="posts", title="posts by subreddit").update_layout(xaxis_tickangle=-30), 360)

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
    return lock(px.scatter(df, x=post_ncom, y=post_score, color=color,
                           labels={post_ncom:"num_comments", post_score:"score"},
                           title="engagement scatter: comments vs score"), 360)

def fig_topics_treemap(td_subset):
    print(td_subset)
    if td_subset is None or td_topic is None: return None
    if td_subset["Name"].nunique() == 1:
        fig = px.treemap(
            td_subset, path=["Name"], values="Count",
            color="Name",
            hover_data=["Representation"] if "Representation" in td_subset.columns else None,
            title="topic landscape (treemap)",
            color_discrete_sequence=["#636EFA"]
        )
    else:
        fig = px.treemap(
            td_subset, path=["Name"], values="Count",
            color="Name",
            hover_data=["Representation"] if "Representation" in td_subset.columns else None,
            title="topic landscape (treemap)",
            color_discrete_sequence=px.colors.qualitative.Vivid  # любая палитра
        )
        fig.update_layout(
            treemapcolorway=px.colors.qualitative.Vivid,
            extendsunburstcolors=True
        )

    return lock(fig, 420)

def fig_topic_engagement(df, td_subset):
    if df is None or td_subset is None or td_topic is None or td_id is None or post_id is None: return None
    merged = df.merge(td_subset[[td_id, td_topic]], left_on=post_id, right_on=td_id, how="inner")
    if merged.empty: return None
    score_col = post_score if (post_score in merged.columns) else None
    ncom_col = post_ncom if (post_ncom in merged.columns) else None
    agg = merged.groupby(td_topic).agg(
        avg_score=(score_col,"mean") if score_col else (td_topic,"size"),
        avg_comments=(ncom_col,"mean") if ncom_col else (td_topic,"size"),
        n=(td_topic,"size"),
    ).reset_index()
    if agg.empty: return None
    agg[td_topic] = agg[td_topic].astype(str)
    agg["TopicLabel"] = agg[td_topic].map(lambda k: topic_label_map.get(k, f"Topic {k}"))
    xcol = "avg_comments" if "avg_comments" in agg.columns else "n"
    ycol = "avg_score" if "avg_score" in agg.columns else "n"
    return lock(px.scatter(agg, x=xcol, y=ycol, size="n", hover_name="TopicLabel",
                           title="topic engagement (avg comments × avg score)"), 360)

def fig_top_topics_bar(td_subset):
    if td_subset is None or td_topic is None: return None
    c = td_subset[td_topic].astype(str).value_counts().head(20).reset_index()
    c.columns = ["topic","freq"]
    c["label"] = c["topic"].map(lambda k: topic_label_map.get(k, f"Topic {k}"))
    return lock(px.bar(c, x="label", y="freq", title="top topics in selection")
                .update_layout(xaxis_tickangle=-45), 360)

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
        style_table={"overflowX":"auto", "maxHeight":"420px", "overflowY":"auto"},
        style_cell={"backgroundColor":"#0f1117","color":"#eaeef2","border":"1px solid #222636","maxWidth":500,"whiteSpace":"nowrap","textOverflow":"ellipsis"},
        style_header={"backgroundColor":"#121317","fontWeight":"bold"},
    )

# app
app = Dash(__name__, title="Reddit Insights Dashboard", suppress_callback_exceptions=True)

# index shell
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
    html, body{{width:100%; height:100%; margin:0; padding:0; overflow-x:hidden}}
    body{{font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background:var(--bg); color:var(--ink);}}
    .header{{padding:24px 16px; position:sticky; top:0; background:linear-gradient(180deg,rgba(11,12,16,.95),rgba(11,12,16,.8)); border-bottom:1px solid #1f2430; z-index:100;}}
    .h1{{font-size:26px; font-weight:800; display:flex; align-items:center; gap:10px}}
    .pill{{font-size:12px; background:linear-gradient(135deg,#a78bfa,#f472b6); padding:6px 10px; border-radius:999px; color:#fff}}
    .sub{{color:var(--muted); margin-top:6px}}
    .container{{max-width:1280px; margin:0 auto; padding:0 12px}}
    .controls{{display:grid; gap:12px; grid-template-columns:1fr 1fr 1fr; padding:12px 0 6px}}
    @media(max-width:980px){{ .controls{{grid-template-columns:1fr}} }}
    select, input[type=text]{{width:100%; padding:10px 12px; border-radius:10px; border:1px solid #222636; background:#0f1117; color:#eaeef2;}}
    .grid{{display:grid; gap:16px; grid-template-columns:repeat(12, 1fr);}}
    .card{{grid-column:span 6; background:linear-gradient(180deg, #161821, #0f1117); border:1px solid #222636; border-radius:16px; padding:12px; overflow:hidden}}
    .card.wide{{grid-column:span 12}}
    .card h2{{font-size:16px; font-weight:700; margin:6px 10px 10px}}
    .footer{{color:#a0a8b7; font-size:12px; text-align:center; padding:18px 0 42px}}
    /* keep plotly canvases inside cards */
    .js-plotly-plot, .dash-graph{{width:100% !important; max-width:100% !important;}}
    </style>
  </head>
  <body>
    <div class="header">
      <div class="container">
        <div class="h1">Reddit Insights Dashboard <span class="pill">interactive • plotly</span></div>
        <div class="sub">only charts supported by your data are shown. no broken cards.</div>
      </div>
    </div>
    <div class="container">
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

# dropdowns
all_subreddits = ["__ALL__"]
if posts is not None and post_subreddit in (posts.columns if posts is not None else []):
    opts = [s for s in posts[post_subreddit].dropna().unique().tolist() if str(s).strip() != ""]
    all_subreddits = ["__ALL__"] + sorted(opts)

# layout
app.layout = html.Div(children=[
    html.Div(className="controls", children=[
        html.Div([
            html.Label("subreddit", style={"color":"#cbd5e1","fontSize":"13px"}),
            dcc.Dropdown(id="subreddit",
                         options=[{"label": s if s!="__ALL__" else "All", "value": s} for s in all_subreddits],
                         value="__ALL__", clearable=False)
        ]),
        html.Div([
            html.Label("topics (multi)", style={"color":"#cbd5e1","fontSize":"13px"}),
            dcc.Dropdown(id="topic-dd", options=TOPIC_OPTIONS, value=[], multi=True, placeholder="select topic(s)")
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

# callback
@app.callback(
    Output("cards", "children"),
    Input("refresh", "n_clicks"),
    Input("subreddit", "value"),
    Input("topic-dd", "value"),
    prevent_initial_call=False
)
def build_cards(n, subreddit_value, selected_topics):
    df_posts = filter_posts(posts, subreddit_value, selected_topics)
    td_subset = filter_topic_dist(selected_topics)
    td_topic_info_subset = filter_topic_info(selected_topics)

    figs = []
    f1 = fig_posts_by_subreddit(df_posts);               f1 and figs.append(("Posts by Subreddit", f1, 360, ""))
    f2 = fig_score_hist(df_posts);                       f2 and figs.append(("Score Distribution", f2, 360, ""))
    f3 = fig_comments_hist(df_posts);                    f3 and figs.append(("Comments Distribution", f3, 360, ""))
    f4 = fig_score_vs_comments(df_posts);                f4 and figs.append(("Engagement Scatter (Comments vs Score)", f4, 360, ""))
    f5 = fig_topics_treemap(td_subset);                  f5 and figs.append(("Topic Landscape (Treemap)", f5, 420, ""))
    f6 = fig_topic_engagement(df_posts, td_subset);      f6 and figs.append(("Topic Engagement", f6, 360, ""))
    f7 = fig_top_topics_bar(td_subset);                  f7 and figs.append(("Top Topics in Selection", f7, 360, ""))

    table = fig_top_posts_table(df_posts, k=20)

    cards = []
    for title, content, h, extra_class in figs:
        cards.append(
            html.Div(className=f"card {extra_class}".strip(), children=[
                html.H2(title),
                dcc.Graph(figure=content, style={"height": f"{h}px", "width": "100%"}, config={"responsive": False, "displaylogo": False})
            ])
        )
    if table is not None:
        cards.append(html.Div(className="card wide", children=[html.H2("Top Posts (by score)"), table]))

    if len(cards) < 3:
        cards.append(html.Div(className="card", children=[html.H2("note"), html.Div("not enough columns for more charts in the current csvs.")]))

    return cards

if __name__ == "__main__":
    app.run(debug=True)
