import os
import streamlit as st
from dotenv import load_dotenv

# Environment must be ready before project imports.
load_dotenv()
try:
    for key in ["S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "GOOGLE_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import numpy as np
import pandas as pd
import plotly.express as px
from ai.build_fact_package import load_raw_history
from analytics.historical_intelligence import prepare_historical_intelligence

st.set_page_config(page_title="YouTube Trend Intelligence", page_icon="▶", layout="wide", initial_sidebar_state="collapsed")

GOLD, GOLD_SOFT = "#D6AE5D", "#E7C77D"
BG, PANEL, BORDER, MUTED, TEXT = "#090A0C", "#101114", "#26282D", "#8B8E98", "#F4F1E8"
EXPECTED_REGIONS = {"CA", "GB", "IN", "US"}

st.markdown(f"""
<style>
:root{{--gold:{GOLD};--bg:{BG};--panel:{PANEL};--border:{BORDER};--muted:{MUTED};--text:{TEXT};}}
.stApp{{background:var(--bg);color:var(--text)}}
.block-container{{max-width:1180px;padding-top:1.5rem;padding-bottom:3rem}}
[data-testid="stHeader"]{{background:rgba(0,0,0,0)}}
[data-testid="stSidebar"]{{background:#08090b;border-right:1px solid var(--border)}}
h1,h2,h3{{letter-spacing:-.025em}}
.gold-kicker{{color:var(--gold);font-size:.76rem;font-weight:700;letter-spacing:.22em}}
.eyebrow{{color:var(--muted);font-size:.72rem;letter-spacing:.18em;text-transform:uppercase}}
.subtle{{color:var(--muted)}}
.hero-title{{font-size:clamp(2.25rem,5vw,4.4rem);line-height:.98;font-weight:700;letter-spacing:-.055em;margin:.6rem 0 1rem;max-width:920px}}
.signal{{border-left:2px solid var(--gold);padding:.2rem 0 .2rem 1.15rem;margin:2rem 0}}
.metric-card{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:1.15rem 1.2rem;min-height:112px}}
.metric-label{{color:var(--muted);font-size:.68rem;letter-spacing:.13em;text-transform:uppercase}}
.metric-value{{color:var(--text);font-size:2rem;font-weight:650;margin-top:.45rem}}
.metric-value.positive{{color:#54C98A}}
.video-title{{font-weight:600;font-size:1rem;line-height:1.25}}
.video-meta{{color:var(--muted);font-size:.78rem;margin-top:.25rem}}
.rank-number{{color:var(--muted);font-size:1.35rem;font-weight:650}}
.rise{{color:#54C98A;font-weight:600}} .fall{{color:#E06C75;font-weight:600}} .neutral{{color:var(--muted);font-weight:600}}
.so-what{{background:#111216;border:1px solid var(--border);border-radius:14px;padding:1.35rem 1.5rem;margin-top:1.5rem}}
hr{{border-color:var(--border)!important}}

    .stButton > button[kind="primary"] {{
        background: #D4A72C !important;
        border-color: #D4A72C !important;
        color: #0b0c0e !important;
        font-weight: 700 !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: #E0B63F !important;
        border-color: #E0B63F !important;
        color: #0b0c0e !important;
    }}
</style>
""", unsafe_allow_html=True)

def safe_text(value, fallback="Unknown"):
    if value is None or pd.isna(value): return fallback
    return str(value)

def fmt_int(value):
    if value is None or pd.isna(value): return "—"
    return f"{int(value):,}"

def fmt_compact(value):
    if value is None or pd.isna(value): return "—"
    value = float(value)
    if abs(value) >= 1_000_000: return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000: return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"

def youtube_thumbnail(video_id):
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else ""

@st.cache_data(ttl=1800, show_spinner=False)
def load_pulse_data():
    # Pulse deliberately avoids Gemini, audit history and ML inference.
    raw = load_raw_history()
    intel = prepare_historical_intelligence(raw)
    return raw, intel

@st.cache_data(ttl=1800, show_spinner=False)
def load_ai_fact_package():
    import json
    import boto3

    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    response = s3.get_object(
        Bucket=os.getenv("S3_BUCKET"),
        Key="serving/trendatlas/ai_fact_package.json",
    )

    return json.loads(
        response["Body"].read().decode("utf-8")
    )

@st.cache_data(ttl=1800, show_spinner=False)
def load_pipeline_audit():
    """Load S3 audit history only when Data Health is opened."""
    import boto3
    from build_history import load_audit_history

    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    audit = load_audit_history(s3)
    if not isinstance(audit, pd.DataFrame):
        audit = pd.DataFrame(audit)
    return audit


def latest_complete_snapshot(latest_state):
    current = latest_state.copy()
    current["observed_date"] = pd.to_datetime(current["fetch_timestamp"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    by_date = current.dropna(subset=["observed_date", "region_code"]).groupby("observed_date")["region_code"].agg(set)
    complete = [d for d, regions in by_date.items() if EXPECTED_REGIONS.issubset(regions)]
    chosen = max(complete) if complete else current["observed_date"].max()
    return current[current["observed_date"] == chosen].copy(), chosen

with st.sidebar:
    st.markdown("### Trend Intelligence")
    st.caption("Story chapters")
    if "chapter" not in st.session_state:
        st.session_state.chapter = "01 · Pulse"
    for item in ["01 · Pulse", "02 · Momentum", "03 · Lifecycle", "04 · Content", "05 · Regions", "06 · Prediction", "07 · AI Brief", "08 · Data Health"]:
        if st.button(item, use_container_width=True,
                     type="primary" if st.session_state.chapter == item else "secondary",
                     key=f"nav_{item}"):
            st.session_state.chapter = item
            st.rerun()
    chapter = st.session_state.chapter
    st.divider()
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# AI Brief builds its own bounded fact package and does not need the
# full dashboard history to be loaded first.
if chapter != "07 · AI Brief":
    try:
        with st.spinner("Loading trend intelligence…"):
            raw, intel = load_pulse_data()
            current, pulse_date = latest_complete_snapshot(
                intel["latest_video_state"]
            )
    except Exception as exc:
        st.error("Trend intelligence could not load the historical snapshots.")
        with st.expander("Technical details"):
            st.exception(exc)
        st.stop()

head_l, head_r = st.columns([5, 1])

with head_l:
    st.markdown(
        '<div class="eyebrow">YouTube Trend Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtle">How attention rises, travels and survives.</div>',
        unsafe_allow_html=True,
    )

with head_r:
    if chapter == "07 · AI Brief":
        st.caption("● Grounded AI brief")
    else:
        st.caption(
            f"● Latest complete view · "
            f"{current['region_code'].nunique()}/4 regions"
        )

st.divider()

if chapter != "07 · AI Brief":
    categories = sorted(
        current["category_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

if chapter == "01 · Pulse":
    st.markdown('<div class="gold-kicker">01 · PULSE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">What is capturing attention right now?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtle">Latest complete snapshot: {pd.to_datetime(pulse_date).date()} across CA, GB, IN and US.</div>', unsafe_allow_html=True)

    category = st.selectbox("Category lens", ["All"] + categories, key="pulse_category", help="This lens will stay consistent across later story chapters.")
    view = current if category == "All" else current[current["category_name"].astype(str) == category]
    if view.empty:
        st.info("No videos are available for this category in the latest complete snapshot.")
        st.stop()

    rank_movement = pd.to_numeric(view["rank_movement"], errors="coerce")
    trending_obs = pd.to_numeric(view["trending_observations"], errors="coerce")
    rising = view[rank_movement > 0].copy()
    new_today = view[trending_obs <= 1].copy() if trending_obs.notna().any() else view.iloc[0:0]

    metrics = [("Live video-region states", len(view)), ("Rising", len(rising)), ("New today", len(new_today)), ("Regions fresh", f"{view['region_code'].nunique()} / 4")]
    for col, (label, value) in zip(st.columns(4), metrics):
        with col:
            shown = fmt_int(value) if isinstance(value, (int, np.integer)) else value
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{shown}</div></div>', unsafe_allow_html=True)

    if not rising.empty:
        rising = rising.sort_values(["rank_movement", "view_velocity"], ascending=[False, False])
        top_riser, move = rising.iloc[0], int(rising.iloc[0]["rank_movement"])
        top_name = safe_text(top_riser.get("video_title"), "The leading video")
    else:
        move, top_name = 0, "No video"
    cat_counts = view["category_name"].dropna().astype(str).value_counts()
    lead_category = cat_counts.index[0] if len(cat_counts) else "No category"
    st.markdown(f'''<div class="signal"><div class="eyebrow">Today's signal</div><div style="font-size:1.35rem;margin-top:.45rem;line-height:1.45"><b>{len(rising):,}</b> video-region states are rising. <b>{top_name}</b> made the largest move, gaining <b>{move}</b> positions, while <b>{lead_category}</b> leads the current mix.</div></div>''', unsafe_allow_html=True)

    st.markdown("### Trending now")
    st.caption("Rank · movement · view velocity · region")
    leaderboard = view.copy()
    leaderboard["trending_rank"] = pd.to_numeric(leaderboard["trending_rank"], errors="coerce")
    leaderboard = leaderboard.sort_values(["trending_rank", "view_velocity"], ascending=[True, False]).head(10)
    for _, row in leaderboard.iterrows():
        c_rank, c_thumb, c_title, c_signal = st.columns([.45, 1.2, 5, 1.7], vertical_alignment="center")
        with c_rank:
            rank = row.get("trending_rank")
            st.markdown(f'<div class="rank-number">#{int(rank) if pd.notna(rank) else "—"}</div>', unsafe_allow_html=True)
        with c_thumb:
            thumb = youtube_thumbnail(safe_text(row.get("id"), ""))
            if thumb:
                st.image(thumb, use_container_width=True)
        with c_title:
            st.markdown(f'<div class="video-title">{safe_text(row.get("video_title"), "Untitled video")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="video-meta">{safe_text(row.get("channel_name"))} · {safe_text(row.get("category_name"))} · {safe_text(row.get("region_code"), "—")}</div>', unsafe_allow_html=True)
        with c_signal:
            movement = row.get("rank_movement")
            if pd.notna(movement) and movement > 0:
                move_html = f'<span class="rise">▲ {int(movement)}</span>'
            elif pd.notna(movement) and movement < 0:
                move_html = f'<span class="fall">▼ {abs(int(movement))}</span>'
            else:
                move_html = '<span class="neutral">→ 0</span>'
            st.markdown(move_html, unsafe_allow_html=True)
            st.caption(f"{fmt_compact(row.get('view_velocity'))} views/hr")
        st.divider()

    left, right = st.columns(2)
    with left:
        mix = view["category_name"].fillna("Unknown").astype(str).value_counts().head(8).rename_axis("category").reset_index(name="states")
        mix_title = "Gaming holds the largest share of today's trending board" if category == "All" else f"{category} activity in today's trending board"
        fig = px.bar(mix.sort_values("states"), x="states", y="category", orientation="h", title=mix_title)
        fig.update_traces(marker_color=GOLD)
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(l=10,r=10,t=55,b=10), height=350, showlegend=False, font=dict(color="#C7C8CD"))
        fig.update_xaxes(title=None,gridcolor="#202227"); fig.update_yaxes(title=None,gridcolor="#202227")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        entry = pd.DataFrame({"status":["New today","Returning"], "states":[len(new_today), max(len(view)-len(new_today),0)]})
        fig = px.bar(entry, x="status", y="states", title="Most trending states are returning, not new")
        fig.update_traces(marker_color=[GOLD,"#555861"])
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(l=10,r=10,t=55,b=10), height=350, showlegend=False, font=dict(color="#C7C8CD"))
        fig.update_xaxes(title=None,gridcolor="#202227"); fig.update_yaxes(title=None,gridcolor="#202227")
        st.plotly_chart(fig, use_container_width=True)

    rising_share = 100 * len(rising) / len(view) if len(view) else 0
    st.markdown(f'''<div class="so-what"><div class="gold-kicker">SO WHAT</div><div style="font-size:1.12rem;margin-top:.5rem">{lead_category} currently has the largest share of collected trending states, and {rising_share:.0f}% of this view is moving upward. The next question is not who is visible — it is <b>who is accelerating fastest.</b></div><div style="color:{GOLD_SOFT};margin-top:.9rem;font-weight:600">Continue → 02 · Momentum</div></div>''', unsafe_allow_html=True)

    with st.expander("How this is calculated"):
        st.markdown("""
**Pulse** uses the latest available date with all four expected regions (CA, GB, IN and US) when one exists.

- **Live video-region states** counts rows in that snapshot after the category lens is applied.
- **Rising** means `rank_movement > 0` — the video improved its trending position relative to its previous observation.
- **New today** uses `trending_observations <= 1` as the current entry proxy.
- **View velocity** is snapshot-based growth per hour; it measures growth observed while the video is in our collected trending history, not the video's full lifetime.
""")

elif chapter == "02 · Momentum":
    st.markdown('<div class="gold-kicker">02 · MOMENTUM</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Which videos are accelerating — and how fast?</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Rank tells us where a video is. Momentum tells us where attention is moving.</div>', unsafe_allow_html=True)

    category = st.selectbox("Category lens", ["All"] + categories, key="momentum_category", help="Filter every Momentum metric and chart to one content category.")
    view = current if category == "All" else current[current["category_name"].astype(str) == category]
    if view.empty:
        st.info("No videos are available for this category in the latest complete snapshot.")
        st.stop()

    m = view.copy()
    m["rank_movement"] = pd.to_numeric(m["rank_movement"], errors="coerce")
    m["trending_rank"] = pd.to_numeric(m["trending_rank"], errors="coerce")
    m["view_velocity"] = pd.to_numeric(m["view_velocity"], errors="coerce")
    m["previous_rank"] = m["trending_rank"] + m["rank_movement"]
    valid_move = m.dropna(subset=["rank_movement"])
    rising = valid_move[valid_move["rank_movement"] > 0].copy()
    fastest = int(rising["rank_movement"].max()) if not rising.empty else 0
    median_move = float(valid_move["rank_movement"].median()) if not valid_move.empty else np.nan
    velocity_rows = m.dropna(subset=["view_velocity"])
    velocity_leader = float(velocity_rows["view_velocity"].max()) if not velocity_rows.empty else np.nan

    metric_specs = [
        ("Rising states", fmt_int(len(rising)), False),
        ("Fastest climb", f"▲ {fastest}" if fastest else "—", fastest > 0),
        ("Median move", (f"▲ {median_move:.0f}" if median_move > 0 else f"▼ {abs(median_move):.0f}" if median_move < 0 else "→ 0") if pd.notna(median_move) else "—", median_move > 0 if pd.notna(median_move) else False),
        ("Velocity leader", f"{fmt_compact(velocity_leader)}/hr" if pd.notna(velocity_leader) else "—", False),
    ]
    for col, (label, value, positive) in zip(st.columns(4), metric_specs):
        with col:
            klass = "metric-value positive" if positive else "metric-value"
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="{klass}">{value}</div></div>', unsafe_allow_html=True)

    if not rising.empty:
        top = rising.sort_values(["rank_movement", "view_velocity"], ascending=[False, False]).iloc[0]
        top_title = safe_text(top.get("video_title"), "The fastest climber")
        top_move = int(top["rank_movement"])
    else:
        top_title, top_move = "No video", 0
    if not velocity_rows.empty:
        vel_top = velocity_rows.sort_values("view_velocity", ascending=False).iloc[0]
        vel_title = safe_text(vel_top.get("video_title"), "The velocity leader")
    else:
        vel_title = "No video"
    same_leader = top_title == vel_title and top_title != "No video"
    relation = "The same video also leads view velocity." if same_leader else "The velocity leader is different, so audience growth and rank movement are not the same signal."
    st.markdown(f'''<div class="signal"><div class="eyebrow">Momentum insight</div><div style="font-size:1.35rem;margin-top:.45rem;line-height:1.45"><b>{top_title}</b> made the fastest climb, gaining <b>{top_move}</b> positions. {relation}</div></div>''', unsafe_allow_html=True)

    st.markdown("### The biggest climbers changed the board quickly")
    st.caption("Previous observed rank → latest rank in the same region. Larger upward jumps mean stronger rank momentum.")
    hero = m.dropna(subset=["previous_rank", "trending_rank", "rank_movement"]).sort_values(["rank_movement", "view_velocity"], ascending=[False, False]).head(6).copy()
    if hero.empty:
        st.info("Not enough prior-rank observations are available for the rank-change view.")
    else:
        for _, row in hero.iterrows():
            a,thumb_col,b,c,d = st.columns([1.25,1.35,4.2,.85,1.35], vertical_alignment="center")
            with a:
                st.markdown(f'<div class="rank-number">#{int(row["previous_rank"])} → #{int(row["trending_rank"])}</div>', unsafe_allow_html=True)
            with thumb_col:
                thumb = youtube_thumbnail(safe_text(row.get("id"), ""))
                if thumb:
                    st.image(thumb, use_container_width=True)
            with b:
                st.markdown(f'<div class="video-title">{safe_text(row.get("video_title"), "Untitled video")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="video-meta">{safe_text(row.get("channel_name"))} · {safe_text(row.get("category_name"))} · {safe_text(row.get("region_code"), "—")}</div>', unsafe_allow_html=True)
            with c:
                mv=row["rank_movement"]
                css="rise" if mv>0 else "fall" if mv<0 else "neutral"
                arrow="▲" if mv>0 else "▼" if mv<0 else "→"
                st.markdown(f'<span class="{css}">{arrow} {abs(int(mv))}</span>', unsafe_allow_html=True)
            with d:
                vv=row.get("view_velocity")
                st.caption(f"{fmt_compact(vv)} views/hr" if pd.notna(vv) else "Velocity unavailable")
            st.divider()

    left, right = st.columns([1.25, .75])
    with left:
        scatter = m.dropna(subset=["view_velocity", "rank_movement"]).copy()
        if scatter.empty:
            st.info("Not enough velocity and rank-movement observations are available for this view.")
        else:
            fig = px.scatter(scatter, x="view_velocity", y="rank_movement", hover_name="video_title", hover_data={"region_code":True,"trending_rank":True,"view_velocity":":,.0f","rank_movement":":+.0f"}, title="Fast growth does not always mean a rank jump")
            fig.update_traces(marker=dict(color=GOLD, size=9, opacity=.82, line=dict(width=.5,color="#0B0C0E")))
            fig.add_hline(y=0, line_dash="dash", line_color="#555861", line_width=1)
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=410, margin=dict(l=10,r=10,t=60,b=10), showlegend=False, font=dict(color="#C7C8CD"))
            fig.update_xaxes(title="View velocity (views/hour, log scale)", gridcolor="#202227", type="log", tickformat="~s")
            fig.update_yaxes(title="Rank movement (positions)", gridcolor="#202227", zeroline=False)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        movers = valid_move[valid_move["rank_movement"] != 0].copy()
        if movers.empty:
            st.info("No rank movers are available in this snapshot.")
        else:
            risers = movers.nlargest(3, "rank_movement")
            fallers = movers.nsmallest(2, "rank_movement")
            movers5 = pd.concat([risers, fallers]).drop_duplicates(subset=["id", "region_code"]).copy()
            movers5["label"] = movers5["video_title"].fillna("Untitled").astype(str).apply(lambda x: x if len(x) <= 36 else x[:33] + "…")
            movers5 = movers5.sort_values("rank_movement")
            colors = ["#E06C75" if x < 0 else "#54C98A" for x in movers5["rank_movement"]]
            fig = px.bar(movers5, x="rank_movement", y="label", orientation="h", title="Biggest rank movers")
            fig.update_traces(marker_color=colors, text=[f"{x:+.0f}" for x in movers5["rank_movement"]], textposition="outside")
            fig.add_vline(x=0, line_color="#555861", line_width=1)
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=410, margin=dict(l=150,r=55,t=60,b=10), showlegend=False, font=dict(color="#C7C8CD"))
            fig.update_xaxes(title="Rank movement (positions)", gridcolor="#202227", range=[min(-45, float(movers5["rank_movement"].min()) - 8), max(50, float(movers5["rank_movement"].max()) + 8)])
            fig.update_yaxes(title=None, gridcolor="#202227")
            st.plotly_chart(fig, use_container_width=True)

    both = m.dropna(subset=["view_velocity", "rank_movement"])
    corr = both["view_velocity"].corr(both["rank_movement"]) if len(both) >= 3 else np.nan
    if pd.isna(corr):
        corr_text = "There are not enough paired observations to summarize the relationship between velocity and rank movement yet."
    elif abs(corr) < .2:
        corr_text = f"View velocity and rank movement show little linear relationship in this snapshot (r = {corr:.2f}). Fast audience growth does not automatically produce a rank jump."
    elif corr > 0:
        corr_text = f"View velocity and rank movement move together to some extent in this snapshot (correlation {corr:.2f}), but the scatter still shows exceptions."
    else:
        corr_text = f"View velocity and rank movement move in opposite directions overall in this snapshot (correlation {corr:.2f}), showing that velocity alone does not explain trending position."
    st.markdown(f'''<div class="so-what"><div class="gold-kicker">SO WHAT</div><div style="font-size:1.12rem;margin-top:.5rem">{corr_text} The next question is whether that momentum <b>survives over time.</b></div><div style="color:{GOLD_SOFT};margin-top:.9rem;font-weight:600">Continue → 03 · Lifecycle</div></div>''', unsafe_allow_html=True)

    with st.expander("How this is calculated"):
        st.markdown("""
**Momentum** uses the same latest complete four-region snapshot as Pulse and applies the category lens before calculating its summaries.

- **Rank movement** compares the latest observed trending rank with the previous observation for that video-region state. Positive values mean the video climbed toward rank #1.
- **Previous rank** is reconstructed as `current rank + rank movement` for the hero comparison.
- **Fastest climb** is the largest positive rank movement in the filtered snapshot.
- **Median move** is the median rank movement across states with a valid movement value.
- **View velocity** is snapshot-based view growth per hour while the video is present in the collected trending history.
- The scatter plot compares view velocity with rank movement. Its correlation is descriptive for the current snapshot; it does not imply causation.
""")


elif chapter == "03 · Lifecycle":
    h = intel["video_history"].copy()

    h["fetch_date"] = pd.to_datetime(
        h["fetch_date"], errors="coerce"
    ).dt.normalize()

    h = h.dropna(
        subset=["id", "region_code", "fetch_date"]
    )

    # =========================================================
    # HEADER
    # =========================================================

    st.markdown(
        '<div class="gold-kicker">03 · LIFECYCLE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">'
        'How long does trending attention survive?'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtle">'
        'Weekly entry cohorts reveal how quickly video-region '
        'lifecycles disappear from the observed trending board.'
        '</div>',
        unsafe_allow_html=True
    )

    # =========================================================
    # FILTERS
    # =========================================================

    categories = sorted(
        h["category_name"]
        .dropna()
        .astype(str)
        .unique()
    )

    f1, f2, f3 = st.columns(3)

    region = f1.selectbox(
        "Region",
        ["All regions", "US", "CA", "GB", "IN"],
        key="life_region"
    )

    category = f2.selectbox(
        "Category",
        ["All categories"] + categories,
        key="life_category"
    )

    cohort_window = f3.selectbox(
        "Cohorts",
        ["Latest 4", "Latest 8", "All"],
        index=0,
        key="life_cohort_window"
    )

    x = h.copy()

    if region != "All regions":
        x = x[x["region_code"] == region]

    if category != "All categories":
        x = x[
            x["category_name"].astype(str) == category
        ]

    # Keep one observation for each video × region × day.
    d = (
        x.sort_values("fetch_timestamp")
        .drop_duplicates(
            ["id", "region_code", "fetch_date"],
            keep="last"
        )
        .copy()
    )

    if d.empty:
        st.info(
            "No lifecycle observations match these filters."
        )

    else:
        # =====================================================
        # LIFECYCLE TABLE
        # =====================================================

        life = (
            d.sort_values(
                ["id", "region_code", "fetch_date"]
            )
            .groupby(
                ["id", "region_code"],
                as_index=False
            )
            .agg(
                first_seen=("fetch_date", "min"),
                last_seen=("fetch_date", "max"),
                observed_days=("fetch_date", "nunique"),
                first_rank=("trending_rank", "first"),
                video_title=("video_title", "last")
            )
        )

        # We deliberately count ACTUAL observed calendar days.
        # We do not assume that every day between first_seen
        # and last_seen was a trending day.
        life["lifetime_days"] = life["observed_days"]

        life["cohort_week"] = (
            life["first_seen"]
            .dt.to_period("W-SUN")
            .apply(lambda p: p.start_time)
        )

        latest_observed_date = d["fetch_date"].max()

        sample_n = len(life)

        # =====================================================
        # RE-ENTRY
        # =====================================================

        seq = d.sort_values(
            ["id", "region_code", "fetch_date"]
        ).copy()

        seq["gap_days"] = (
            seq.groupby(
                ["id", "region_code"]
            )["fetch_date"]
            .diff()
            .dt.days
        )

        reentered = (
            seq.loc[
                seq["gap_days"] > 1,
                ["id", "region_code"]
            ]
            .drop_duplicates()
        )

        reentry_rate = (
            len(reentered) / sample_n
            if sample_n
            else 0
        )

        # =====================================================
        # KPI METRICS
        # =====================================================

        median_life = life["lifetime_days"].median()

        survive_two = (
            life["lifetime_days"] >= 2
        ).mean()

        longest_life = int(
            life["lifetime_days"].max()
        )

        st.caption(
            f"{sample_n:,} video-region lifecycles "
            "in this selection."
        )
        if category != "All categories":
            st.caption(
                "ⓘ Category labels come directly from YouTube's "
                "video metadata and are not inferred from video titles. "
                "A video's category may therefore appear different from "
                "what its title or subject might suggest."
            )

        metrics = [
            (
                "Median observed life",
                f"{median_life:.0f} days"
            ),
            (
                "Observed 2+ days",
                f"{survive_two:.0%}"
            ),
            (
                "Longest observed",
                f"{longest_life} days"
            ),
            (
                "Re-entry rate",
                f"{reentry_rate:.0%}"
            ),
        ]

        for col, (label, value) in zip(
            st.columns(4),
            metrics
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            {label}
                        </div>
                        <div class="metric-value">
                            {value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # =====================================================
        # SURVIVAL PATTERN
        # =====================================================

        if sample_n < 10:
            survival_message = (
                f"This lens contains only "
                f"<b>{sample_n}</b> video-region lifecycles. "
                "Treat the pattern as descriptive rather than "
                "a reliable general trend."
            )
        else:
            survival_message = (
                f"The typical video-region lifecycle was observed "
                f"on <b>{median_life:.0f} calendar day"
                f"{'' if median_life == 1 else 's'}</b>; "
                f"<b>{survive_two:.0%}</b> were observed on "
                "at least two distinct days."
            )

        st.markdown(
            f"""
            <div class="signal">
                <div class="eyebrow">
                    SURVIVAL PATTERN
                </div>
                <div style="
                    font-size:1.35rem;
                    margin-top:.45rem;
                    line-height:1.45;
                ">
                    {survival_message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =====================================================
        # GRAPH 1 — WEEKLY COHORT SURVIVAL
        # =====================================================

        st.markdown(
            "### Trending attention falls sharply after entry"
        )

        st.caption(
            "Each line represents video-region lifecycles that "
            "first entered the observed trending history during "
            "the same week. Hover over a point to inspect the "
            "survivors behind it."
        )

        survival_obs = d[
            [
                "id",
                "region_code",
                "fetch_date",
                "trending_rank",
                "video_title",
                "view_count"
            ]
        ].merge(
            life[
                [
                    "id",
                    "region_code",
                    "first_seen",
                    "cohort_week"
                ]
            ],
            on=["id", "region_code"],
            how="left"
        )

        survival_obs["day_since_entry"] = (
            survival_obs["fetch_date"]
            - survival_obs["first_seen"]
        ).dt.days

        all_cohorts = sorted(
            pd.to_datetime(
                life["cohort_week"]
                .dropna()
                .unique()
            )
        )

        # For Latest 4 / Latest 8, only compare cohorts that
        # have had at least one full day of follow-up.
        mature_cohorts = [
            cohort
            for cohort in all_cohorts
            if (
                latest_observed_date
                - pd.Timestamp(cohort)
            ).days >= 7
        ]

        if cohort_window == "Latest 4":
            selected_cohorts = mature_cohorts[-4:]

        elif cohort_window == "Latest 8":
            selected_cohorts = mature_cohorts[-8:]

        else:
            selected_cohorts = all_cohorts

        survival_rows = []

        # First week after entry.
        max_survival_day = 6

        for cohort in selected_cohorts:

            cohort = pd.Timestamp(cohort)

            cohort_life = life[
                life["cohort_week"] == cohort
            ].copy()

            cohort_obs = survival_obs[
                survival_obs["cohort_week"] == cohort
            ].copy()

            if cohort_life.empty:
                continue

            for day in range(
                0,
                max_survival_day + 1
            ):

                # Right-censoring protection:
                # only include states that have had enough time
                # in the dataset to reach this day.
                eligible = cohort_life[
                    (
                        latest_observed_date
                        - cohort_life["first_seen"]
                    ).dt.days >= day
                ].copy()

                eligible_n = len(eligible)

                if eligible_n == 0:
                    continue

                day_obs = cohort_obs[
                    cohort_obs["day_since_entry"] == day
                ].copy()

                eligible_keys = set(
                    zip(
                        eligible["id"],
                        eligible["region_code"]
                    )
                )

                day_obs = day_obs[
                    day_obs.apply(
                        lambda row: (
                            row["id"],
                            row["region_code"]
                        ) in eligible_keys,
                        axis=1
                    )
                ]

                day_obs = day_obs.drop_duplicates(
                    ["id", "region_code"]
                )

                survivor_n = len(day_obs)

                survival_pct = (
                    100 * survivor_n / eligible_n
                )

                # ---------------------------------------------
                # Example survivors for hover
                # ---------------------------------------------

                examples = []

                                # Pick the available rank column safely.
                rank_col = next(
                    (
                        col for col in [
                            "trending_rank",
                            "trending_rank_x",
                            "trending_rank_y"
                        ]
                        if col in day_obs.columns
                    ),
                    None
                )

                if rank_col is not None:
                    example_rows = (
                        day_obs.sort_values(
                            rank_col,
                            ascending=True,
                            na_position="last"
                        )
                        .head(3)
                    )
                else:
                    example_rows = day_obs.head(3)

                for _, ex in example_rows.iterrows():

                    title = str(
                        ex.get(
                            "video_title",
                            "Unknown video"
                        )
                    )

                    # Keep tooltip readable.
                    if len(title) > 48:
                        title = title[:45] + "..."

                    ex_region = str(
                        ex.get(
                            "region_code",
                            ""
                        )
                    )

                    rank = pd.to_numeric(
                        ex.get(rank_col)
                        if rank_col is not None
                        else None,
                        errors="coerce"
                    )

                    views = pd.to_numeric(
                        ex.get("view_count"),
                        errors="coerce"
                    )

                    rank_text = (
                        f"#{int(rank)}"
                        if pd.notna(rank)
                        else "N/A"
                    )

                    if pd.notna(views):
                        if views >= 1_000_000:
                            views_text = (
                                f"{views / 1_000_000:.1f}M"
                            )
                        elif views >= 1_000:
                            views_text = (
                                f"{views / 1_000:.0f}K"
                            )
                        else:
                            views_text = (
                                f"{views:,.0f}"
                            )
                    else:
                        views_text = "N/A"

                    examples.append(
                        f"{title} · {ex_region} · "
                        f"Rank {rank_text} · "
                        f"{views_text} views"
                    )

                if examples:
                    example_text = "<br>".join(
                        examples
                    )
                else:
                    example_text = (
                        "No surviving examples"
                    )

                survival_rows.append(
                    {
                        "Day since entry": day,
                        "Still trending (%)":
                            survival_pct,
                        "Cohort":
                            cohort.strftime("%b %d"),
                        "Survivors":
                            survivor_n,
                        "Eligible":
                            eligible_n,
                        "Examples":
                            example_text,
                    }
                )

        if survival_rows:

            survival_df = pd.DataFrame(
                survival_rows
            )

            fig = px.line(
                survival_df,
                x="Day since entry",
                y="Still trending (%)",
                color="Cohort",
                markers=True,
                custom_data=[
                    "Survivors",
                    "Eligible",
                    "Examples"
                ]
            )

            fig.update_traces(
                hovertemplate=(
                    "<b>Week of %{fullData.name}</b><br>"
                    "Day +%{x}<br><br>"
                    "<b>%{y:.1f}% still trending</b><br>"
                    "%{customdata[0]:,.0f} survivors "
                    "of %{customdata[1]:,.0f} eligible"
                    "<br><br>"
                    "<b>Example survivors</b><br>"
                    "%{customdata[2]}"
                    "<extra></extra>"
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                height=410,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),
                legend_title_text="Week of"
            )

            fig.update_xaxes(
                dtick=1,
                title="Days since entry",
                gridcolor="#202227"
            )

            fig.update_yaxes(
                range=[0, 105],
                ticksuffix="%",
                title="Still trending",
                gridcolor="#202227"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "Not enough follow-up history is "
                "available for these cohorts."
            )

        # =====================================================
        # GRAPH 2 + GRAPH 3
        # =====================================================

        left, right = st.columns(2)

        # =====================================================
        # GRAPH 2 — LIFETIME DISTRIBUTION
        # =====================================================

        with left:

            st.markdown(
                "### Most trending lifecycles last only 1–2 days"
            )

            st.caption(
                "A video-region lifecycle means one video "
                "trending in one region. Durations of 7 or more "
                "observed days are grouped into 7+. Hover over "
                "a bar to inspect examples from that lifetime."
            )

            life_for_dist = life.copy()

            life_for_dist["duration_bucket"] = (
                life_for_dist["lifetime_days"]
                .clip(upper=7)
                .astype(int)
            )

            total_lifecycles = len(
                life_for_dist
            )

            dist_rows = []

            for day in range(1, 8):

                bucket = life_for_dist[
                    life_for_dist[
                        "duration_bucket"
                    ] == day
                ].copy()

                count = len(bucket)

                if count == 0:
                    continue

                share = (
                    100 * count / total_lifecycles
                    if total_lifecycles
                    else 0
                )

                examples = []

                for _, ex in (
                    bucket
                    .sort_values(
                        "first_rank",
                        ascending=True
                    )
                    .head(3)
                    .iterrows()
                ):

                    title = str(
                        ex.get(
                            "video_title",
                            "Unknown video"
                        )
                    )

                    if len(title) > 45:
                        title = title[:42] + "..."

                    ex_region = str(
                        ex.get(
                            "region_code",
                            ""
                        )
                    )

                    examples.append(
                        f"{title} · {ex_region}"
                    )

                example_text = (
                    "<br>".join(examples)
                    if examples
                    else "No examples"
                )

                label = (
                    "7+ days"
                    if day == 7
                    else (
                        "1 day"
                        if day == 1
                        else f"{day} days"
                    )
                )

                dist_rows.append(
                    {
                        "Lifetime": label,
                        "Video-region lifecycles":
                            count,
                        "Share": share,
                        "Examples":
                            example_text
                    }
                )

            dist = pd.DataFrame(
                dist_rows
            )

            fig = px.bar(
                dist,
                x="Lifetime",
                y="Video-region lifecycles",
                custom_data=[
                    "Share",
                    "Examples"
                ]
            )

            fig.update_traces(
                marker_color=GOLD,
                hovertemplate=(
                    "<b>%{x}</b><br><br>"
                    "%{y:,.0f} video-region lifecycles<br>"
                    "%{customdata[0]:.1f}% of "
                    "lifecycles in this lens"
                    "<br><br>"
                    "<b>Examples</b><br>"
                    "%{customdata[1]}"
                    "<extra></extra>"
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                height=350,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),
                showlegend=False
            )

            fig.update_xaxes(
                title=None
            )

            fig.update_yaxes(
                title="Video-region lifecycles",
                gridcolor="#202227"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =====================================================
        # GRAPH 3 — ENTRY RANK VS LIFETIME
        # =====================================================

        with right:

            st.markdown(
                "### Does entry rank relate to trending lifetime?"
            )

            st.caption(
                "Median observed lifetime by the rank where "
                "each video-region lifecycle first entered "
                "our collected trending history."
            )

            rank_numeric = pd.to_numeric(
                life["first_rank"],
                errors="coerce"
            )

            bands = pd.cut(
                rank_numeric,
                bins=[0, 10, 50, 100],
                labels=[
                    "Top 10",
                    "Rank 11–50",
                    "Rank 51–100"
                ]
            )

            ranked_life = life.assign(
                entry_band=bands
            )

            rank_summary = (
                ranked_life
                .dropna(
                    subset=["entry_band"]
                )
                .groupby(
                    "entry_band",
                    observed=False
                )
                .agg(
                    median_days=(
                        "lifetime_days",
                        "median"
                    ),
                    sample_size=(
                        "id",
                        "size"
                    )
                )
                .reset_index()
            )

            rank_summary["entry_band"] = (
                rank_summary[
                    "entry_band"
                ].astype(str)
            )

            # Determine whether there is enough evidence
            # for a useful comparison.
            enough_bands = (
                rank_summary[
                    "sample_size"
                ] >= 5
            )

            if (
                sample_n < 10
                or enough_bands.sum() < 2
            ):

                st.info(
                    "Not enough observations for a reliable "
                    "entry-rank comparison under this lens."
                )

                for _, row in (
                    rank_summary.iterrows()
                ):

                    n = int(
                        row["sample_size"]
                    )

                    if n > 0:
                        st.markdown(
                            f"**{row['entry_band']}** — "
                            f"{row['median_days']:.1f} median "
                            f"observed days · n={n:,}"
                        )
                    else:
                        st.markdown(
                            f"**{row['entry_band']}** — "
                            "No observations"
                        )

            else:

                # Build examples for each rank band.
                rank_examples = {}

                for band in [
                    "Top 10",
                    "Rank 11–50",
                    "Rank 51–100"
                ]:

                    bucket = ranked_life[
                        ranked_life[
                            "entry_band"
                        ].astype(str) == band
                    ].copy()

                    examples = []

                    for _, ex in (
                        bucket
                        .sort_values(
                            "first_rank"
                        )
                        .head(3)
                        .iterrows()
                    ):

                        title = str(
                            ex.get(
                                "video_title",
                                "Unknown video"
                            )
                        )

                        if len(title) > 42:
                            title = (
                                title[:39] + "..."
                            )

                        ex_region = str(
                            ex.get(
                                "region_code",
                                ""
                            )
                        )

                        rank = pd.to_numeric(
                            ex.get(
                                "first_rank"
                            ),
                            errors="coerce"
                        )

                        rank_text = (
                            f"#{int(rank)}"
                            if pd.notna(rank)
                            else "N/A"
                        )

                        days = int(
                            ex.get(
                                "lifetime_days",
                                0
                            )
                        )

                        examples.append(
                            f"{title} · {ex_region} · "
                            f"entered {rank_text} · "
                            f"{days} day"
                            f"{'' if days == 1 else 's'}"
                        )

                    rank_examples[band] = (
                        "<br>".join(examples)
                        if examples
                        else "No examples"
                    )

                rank_summary["Examples"] = (
                    rank_summary[
                        "entry_band"
                    ].map(rank_examples)
                )

                fig = px.bar(
                    rank_summary,
                    x="median_days",
                    y="entry_band",
                    orientation="h",
                    text="median_days",
                    custom_data=[
                        "sample_size",
                        "Examples"
                    ],
                    category_orders={
                        "entry_band": [
                            "Rank 51–100",
                            "Rank 11–50",
                            "Top 10"
                        ]
                    }
                )

                fig.update_traces(
                    marker_color=GOLD,
                    texttemplate=(
                        "%{text:.1f} days"
                    ),
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br><br>"
                        "Median observed lifetime: "
                        "%{x:.1f} days<br>"
                        "Video-region lifecycles: "
                        "%{customdata[0]:,.0f}"
                        "<br><br>"
                        "<b>Example entries</b><br>"
                        "%{customdata[1]}"
                        "<extra></extra>"
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=BG,
                    plot_bgcolor=BG,
                    height=350,
                    margin=dict(
                        l=10,
                        r=70,
                        t=10,
                        b=10
                    ),
                    showlegend=False,
                    yaxis_title=None
                )

                fig.update_xaxes(
                    title="Median observed lifetime (days)",
                    gridcolor="#202227"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # =====================================================
        # SO WHAT
        # =====================================================


        st.markdown(
                f"""
    <div class="so-what">
    <div class="gold-kicker">SO WHAT</div>
    <div style="font-size:1.12rem; margin-top:.5rem;">
    Trending attention decays quickly after entry, with most observed
    video-region lifecycles lasting only 1–2 days. In this dataset,
    higher-ranked entries also show longer median observed lifetimes.
    The next question is <b>what kind of content keeps winning?</b>
    </div>
    <div style="color:{GOLD_SOFT}; margin-top:.9rem; font-weight:600;">
    Continue → 04 · Content
    </div>
    </div>
    """,
                unsafe_allow_html=True
            )

         
        # =====================================================
        # METHODOLOGY
        # =====================================================

        with st.expander(
            "How this is calculated"
        ):
            st.markdown(
                """
**Lifecycle unit:** one video × region.

- **Entry** is the first calendar day a video-region state appears in the collected trending history.
- **Observed life** counts the distinct calendar days on which that video-region state was actually observed. Missing calendar days are not automatically treated as trending days.
- **Weekly entry cohorts** group video-region states by the week of their first observed entry.
- **Day +N survival** requires an actual observation exactly N calendar days after entry.
- Video-region states that have not yet had enough follow-up time to reach Day +N are excluded from that day's denominator.
- **Entry rank** is the trending rank at the first observation of that video-region lifecycle.
- **Re-entry** means the same video-region state appears again after a gap of more than one calendar day.
- Category labels come from YouTube source metadata: the video's `snippet.categoryId` is mapped through YouTube's video-category metadata. Categories are not inferred from titles.

Hover examples are provided to connect aggregate statistics to the underlying videos; they are examples rather than a complete list of records.

All lifecycle measurements describe the project's collected observation window and should not be interpreted as a video's complete lifetime on YouTube.
"""
            )


elif chapter == "04 · Content":
    # ---------------------------------------------------------
    # 04 · CONTENT
    # Question:
    # What kind of content wins — and keeps winning?
    # ---------------------------------------------------------

    h = intel["video_history"].copy()
    h["fetch_date"] = pd.to_datetime(
        h["fetch_date"], errors="coerce"
    ).dt.normalize()

    h = h.dropna(subset=["fetch_date", "category_name"])

    st.markdown(
        '<div class="gold-kicker">04 · CONTENT</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-title">What kind of content wins — and keeps winning?</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtle">'
        'Compare category share, audience velocity and creator concentration '
        'across the collected markets.'
        '</div>',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------

    f1, f2 = st.columns(2)

    grain = f1.selectbox(
        "Time grain",
        ["Weekly", "Daily"],
        key="content_grain"
    )

    region = f2.selectbox(
        "Region",
        ["All regions", "US", "CA", "GB", "IN"],
        key="content_region"
    )

    # Region controls the analytical lens.
    if region == "All regions":
        x = h.copy()
        region_label = "the four collected markets"
    else:
        x = h[h["region_code"] == region].copy()
        region_label = region

    # Treat blank / Unknown categories as missing metadata for
    # category-level comparisons.
    valid_category = (
        x["category_name"]
        .astype(str)
        .str.strip()
        .ne("")
        &
        x["category_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("unknown")
    )

    category_x = x[valid_category].copy()

    if category_x.empty:
        st.warning(
            "No categorized trending observations are available "
            "for this selection."
        )
        st.stop()

    # ---------------------------------------------------------
    # CATEGORY KPIs
    # ---------------------------------------------------------

    counts = category_x["category_name"].value_counts()

    lead = str(counts.index[0])
    lead_share = float(counts.iloc[0] / counts.sum())

    perf = (
        category_x
        .groupby("category_name")
        .agg(
            median_velocity=("view_velocity", "median"),
            median_engagement=("engagement_rate", "median"),
            observations=("id", "count"),
        )
        .reset_index()
    )

    velocity_perf = (
        perf
        .dropna(subset=["median_velocity"])
        .sort_values("median_velocity", ascending=False)
    )

    velocity_leader = (
        str(velocity_perf.iloc[0]["category_name"])
        if not velocity_perf.empty
        else "—"
    )

    velocity_leader_value = (
        float(velocity_perf.iloc[0]["median_velocity"])
        if not velocity_perf.empty
        else None
    )

    # ---------------------------------------------------------
    # CHANNEL CONCENTRATION
    # ---------------------------------------------------------

    # Prefer a stable channel ID if it actually exists in the
    # historical dataset. Otherwise fall back to channel name.
    if (
        "channel_id" in x.columns
        and x["channel_id"].notna().any()
    ):
        channel_key = "channel_id"
        channel_method = "channel ID"
    else:
        channel_key = "channel_name"
        channel_method = "channel name"

    channel_data = x.dropna(subset=[channel_key]).copy()

    if not channel_data.empty:
        cc = (
            channel_data
            .groupby(channel_key)
            .size()
            .sort_values(ascending=False)
        )

        total_channel_obs = cc.sum()

        top10 = (
            float(cc.head(10).sum() / total_channel_obs)
            if total_channel_obs
            else 0.0
        )

        top1 = (
            float(cc.head(1).sum() / total_channel_obs)
            if total_channel_obs
            else 0.0
        )

        unique_channels = int(cc.size)

    else:
        cc = pd.Series(dtype=float)
        top10 = 0.0
        top1 = 0.0
        unique_channels = 0

    # ---------------------------------------------------------
    # KPI CARDS
    # ---------------------------------------------------------

    velocity_value_text = (
        f"{velocity_leader}"
        if velocity_leader_value is not None
        else "—"
    )

    kpis = [
        ("Leading category", lead),
        ("Share of trending", f"{lead_share:.0%}"),
        ("View velocity leader", velocity_value_text),
        ("Top-10 channel share", f"{top10:.0%}"),
    ]

    for col, (label, value) in zip(st.columns(4), kpis):
        with col:
            st.markdown(
                f'''
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    # ---------------------------------------------------------
    # CONTENT PATTERN
    # ---------------------------------------------------------

    if velocity_leader_value is not None:
        velocity_sentence = (
            f"<b>{velocity_leader}</b> has the highest median "
            f"observed view velocity in this selection "
            f"({velocity_leader_value:,.0f} views/hour)."
        )
    else:
        velocity_sentence = ""

    st.markdown(
        f'''
        <div class="signal">
            <div class="eyebrow">Content pattern</div>
            <div style="font-size:1.35rem;margin-top:.45rem;line-height:1.45">
                In <b>{region_label}</b>, <b>{lead}</b> accounts for the
                largest share of categorized trending observations at
                <b>{lead_share:.0%}</b>. {velocity_sentence}
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # CATEGORY MIX OVER TIME
    # ---------------------------------------------------------

    st.markdown("### The content mix changes as attention rotates")

    if grain == "Weekly":
        st.caption(
            "Weekly view groups observations into calendar-week entry periods "
            "for a clearer view of category rotation."
        )
    else:
        st.caption(
            "Daily view shows the same selected market at day-level resolution."
        )

    tmp = category_x.copy()

    if grain == "Daily":
        tmp["period"] = tmp["fetch_date"]
    else:
        tmp["period"] = (
            tmp["fetch_date"]
            .dt.to_period("W-SUN")
            .apply(lambda z: z.start_time)
        )

    mix = (
        tmp
        .groupby(["period", "category_name"])
        .size()
        .rename("observations")
        .reset_index()
    )

    mix["total"] = (
        mix
        .groupby("period")["observations"]
        .transform("sum")
    )

    mix["share"] = (
        100 * mix["observations"] / mix["total"]
    )

    # Keep the chart readable by showing the six largest
    # categories within the CURRENT REGION SELECTION.
    top_categories = counts.head(6).index.tolist()

    mix = mix[
        mix["category_name"].isin(top_categories)
    ].copy()

    if not mix.empty:
        fig = px.line(
            mix,
            x="period",
            y="share",
            color="category_name",
            markers=True,
            custom_data=[
                "category_name",
                "observations"
            ],
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Period: %{x|%b %d, %Y}<br>"
                "Share of categorized trending observations: "
                "%{y:.1f}%<br>"
                "Observations: %{customdata[1]:,.0f}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=390,
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="Category",
        )

        fig.update_yaxes(
            title="Share of categorized trending observations",
            ticksuffix="%"
        )

        fig.update_xaxes(title=None)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------------------------
    # CATEGORY PERFORMANCE + CHANNEL CONCENTRATION
    # ---------------------------------------------------------

    a, b = st.columns([3, 2])

    with a:
        st.markdown("### Categories convert attention differently")

        st.caption(
            "Each bubble represents a YouTube category. Position compares "
            "median observed view velocity with median engagement; bubble "
            "size reflects the number of collected observations."
        )

        bubble = perf.dropna(
            subset=["median_velocity", "median_engagement"]
        ).copy()

        bubble["engagement_pct"] = (
            100 * bubble["median_engagement"]
        )

        if not bubble.empty:

            # Label only the most represented categories to avoid
            # turning the chart into a wall of overlapping text.
            label_categories = (
                bubble
                .nlargest(min(8, len(bubble)), "observations")
                ["category_name"]
                .tolist()
            )

            bubble["display_label"] = bubble["category_name"].where(
                bubble["category_name"].isin(label_categories),
                ""
            )

            fig = px.scatter(
                bubble,
                x="median_velocity",
                y="engagement_pct",
                size="observations",
                text="display_label",
                custom_data=[
                    "category_name",
                    "observations"
                ],
                size_max=36,
            )

            fig.update_traces(
                marker_color=GOLD,
                textposition="top center",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Median view velocity: %{x:,.0f} views/hour<br>"
                    "Median engagement rate: %{y:.2f}%<br>"
                    "Collected observations: %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                height=400,
                margin=dict(l=10, r=10, t=15, b=10),
                showlegend=False,
            )

            fig.update_xaxes(
                title="Median observed view velocity (views/hour)"
            )

            fig.update_yaxes(
                title="Median engagement rate",
                ticksuffix="%"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "Not enough enriched observations are available "
                "for the category performance comparison."
            )

    with b:
        st.markdown(
            "### Attention is spread beyond the largest channels"
        )

        st.metric(
            "Top 10 channel share",
            f"{top10:.0%}"
        )

        st.metric(
            "Unique channels",
            fmt_int(unique_channels)
        )

        st.metric(
            "#1 channel share",
            f"{top1:.1%}"
        )

        st.caption(
            f"Creator concentration is calculated using "
            f"{channel_method} in the available history."
        )

    # ---------------------------------------------------------
    # SO WHAT
    # ---------------------------------------------------------

    if region == "All regions":
        so_what_text = (
            f"<b>{lead}</b> leads the collected category mix across the "
            f"four markets, while <b>{velocity_leader}</b> leads median "
            f"observed view velocity. But an aggregate winner can hide "
            f"large market differences. <b>Does trending attention travel "
            f"across countries?</b>"
        )
    else:
        so_what_text = (
            f"In <b>{region}</b>, <b>{lead}</b> captures the largest share "
            f"of categorized trending observations, while "
            f"<b>{velocity_leader}</b> leads median observed view velocity. "
            f"The next question is whether those patterns remain local or "
            f"also appear in other markets. <b>Does trending travel?</b>"
        )

    st.markdown(
        f'''
        <div class="so-what">
            <div class="gold-kicker">SO WHAT</div>
            <div style="font-size:1.12rem;margin-top:.5rem;line-height:1.55">
                {so_what_text}
            </div>
            <div style="color:{GOLD_SOFT};margin-top:.9rem;font-weight:600">
                Continue → 05 · Regions
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # METHODOLOGY
    # ---------------------------------------------------------

    with st.expander("How this is calculated"):
        st.markdown(
            f"""
**Category source.** Categories come from the category metadata supplied
through the YouTube Data API during ingestion. The dashboard does not infer
a video's category from its title or description.

**Unknown categories.** Records whose stored category is `Unknown` are
excluded from category rankings, category-share charts and category
performance comparisons because they do not identify a usable content
category.

**Category share.** “Share of trending” means a category's share of the
categorized trending observations collected by this pipeline under the
selected region lens. It does **not** represent the category's share of all
videos or all viewing activity on YouTube.

**Time grain.** Weekly and Daily change the resolution of the category-mix
trend chart. The headline KPI cards summarize the full available history
under the selected region lens.

**View velocity.** View velocity is based on changes observed between
historical snapshots in this dataset and is expressed as views per hour.
It is an observed snapshot-growth measure, not a video's lifetime average
view rate.

**Engagement.** The category comparison uses the median engagement rate
available in the historical intelligence data. Medians are used to reduce
the influence of extreme observations.

**Bubble size.** Bubble size represents the number of categorized
video-region observations available for that category under the selected
region lens.

**Creator concentration.** Top-10 and #1 channel share are calculated from
the collected trending observations under the selected region lens.
This run uses **{channel_method}** as the creator identifier.

**Scope.** Results describe this project's collected YouTube trending
snapshots for CA, GB, IN and US. They should not be interpreted as a
measurement of the entire YouTube platform.
"""
        )


elif chapter == "05 · Regions":
    h = intel["video_history"].copy()
    h["fetch_date"] = pd.to_datetime(
        h["fetch_date"], errors="coerce"
    ).dt.normalize()

    h = h.dropna(subset=["fetch_date", "region_code", "id"])

    st.markdown(
        '<div class="gold-kicker">05 · REGIONS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">'
        'Does trending attention travel across countries?'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtle">'
        'Compare market-specific content with videos that cross CA, GB, IN and US.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Filters
    # ---------------------------------------------------------
    f1, f2 = st.columns(2)

    window = f1.selectbox(
        "Time window",
        [
            "Full history",
            "Last 4 weeks",
            "Last 2 weeks",
            "Latest week",
        ],
        key="region_window",
    )

    cats = sorted(
        h["category_name"]
        .dropna()
        .astype(str)
        .unique()
    )

    category = f2.selectbox(
        "Category",
        ["All categories"] + cats,
        key="region_category",
    )

    # ---------------------------------------------------------
    # Apply TIME filter first.
    #
    # Important:
    # base_window keeps ALL categories inside the selected
    # time period. We need this later so category share is
    # calculated against the whole trending board.
    # ---------------------------------------------------------
    max_date = h["fetch_date"].max()

    ndays = {
        "Full history": None,
        "Last 4 weeks": 28,
        "Last 2 weeks": 14,
        "Latest week": 7,
    }[window]

    if ndays is None:
        base_window = h.copy()
    else:
        start_date = max_date - pd.Timedelta(days=ndays - 1)
        base_window = h[h["fetch_date"] >= start_date].copy()

    # x is the selected analytical lens.
    if category == "All categories":
        x = base_window.copy()
    else:
        x = base_window[
            base_window["category_name"].astype(str) == category
        ].copy()

    # ---------------------------------------------------------
    # Geographic reach
    # ---------------------------------------------------------
    reach = (
        x.groupby("id")["region_code"]
        .nunique()
        .rename("markets_trending")
    )

    cross_share = (
        (reach >= 2).mean()
        if len(reach)
        else np.nan
    )

    all4 = (
        (reach >= 4).mean()
        if len(reach)
        else np.nan
    )

    # ---------------------------------------------------------
    # Local vs cross-market velocity
    # ---------------------------------------------------------
    rr = x.merge(
        reach,
        on="id",
        how="left",
    )

    rr["reach_type"] = np.where(
        rr["markets_trending"] >= 2,
        "Cross-region",
        "Local only",
    )

    med = (
        rr.groupby("reach_type")["view_velocity"]
        .median()
    )

    lv = med.get("Local only", np.nan)
    cv = med.get("Cross-region", np.nan)

    lift = (
        cv / lv
        if (
            pd.notna(lv)
            and lv > 0
            and pd.notna(cv)
        )
        else np.nan
    )

    # ---------------------------------------------------------
    # TRUE within-market category composition
    #
    # Denominator = ALL categorized trending observations
    # in that market during the selected time window.
    #
    # This is intentionally calculated BEFORE applying the
    # category filter. Therefore selecting Music does NOT
    # make Music equal 100% automatically.
    # ---------------------------------------------------------
    categorized_base = base_window.dropna(
        subset=["category_name"]
    ).copy()

    comp = (
        categorized_base
        .groupby(["region_code", "category_name"])
        .size()
        .rename("n")
        .reset_index()
    )

    region_totals = (
        categorized_base
        .groupby("region_code")
        .size()
        .rename("region_total")
        .reset_index()
    )

    comp = comp.merge(
        region_totals,
        on="region_code",
        how="left",
    )

    comp["share"] = (
        comp["n"] / comp["region_total"]
    )

    # ---------------------------------------------------------
    # Most locally distinct market
    #
    # Compare each market's full category-share distribution
    # with the overall category distribution in this time
    # window using total variation distance.
    #
    # This intentionally remains a market-composition measure,
    # even when a single category is selected.
    # ---------------------------------------------------------
    overall = (
        categorized_base["category_name"]
        .value_counts(normalize=True)
    )

    distinct = {}

    for reg, grp in comp.groupby("region_code"):
        region_share = grp.set_index(
            "category_name"
        )["share"]

        idx = overall.index.union(
            region_share.index
        )

        distance = (
            region_share.reindex(
                idx,
                fill_value=0,
            )
            - overall.reindex(
                idx,
                fill_value=0,
            )
        ).abs().sum() / 2

        distinct[reg] = distance

    most_distinct = (
        max(distinct, key=distinct.get)
        if distinct
        else "—"
    )

    # ---------------------------------------------------------
    # KPI cards
    # ---------------------------------------------------------
    cross_display = (
        f"{cross_share:.0%}"
        if pd.notna(cross_share)
        else "—"
    )

    all4_display = (
        f"{all4:.0%}"
        if pd.notna(all4)
        else "—"
    )

    lift_display = (
    f"{lift:.1f}×"
    if pd.notna(lift)
    else "—"
    )

    kpis = [
        (
            "Cross-market videos",
            cross_display,
        ),
        (
            "All 4 markets",
            all4_display,
        ),
        (
            "Most locally distinct mix",
            most_distinct,
        ),
        (
            "Cross-region velocity",
            lift_display,
        ),
    ]

    for col, (label, value) in zip(
        st.columns(4),
        kpis,
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {label}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # Geographic signal
    # ---------------------------------------------------------
    if pd.notna(cross_share):
        signal_text = (
            f"<b>{cross_share:.0%}</b> of unique videos in this lens "
            f"appear in at least two collected markets"
        )

        if pd.notna(all4):
            signal_text += (
                f", while only <b>{all4:.0%}</b> reach all four"
            )

        signal_text += "."

        if pd.notna(lift):
            signal_text += (
                f" Cross-region videos show "
                f"<b>{lift:.1f}×</b> the median observed view velocity "
                f"of local-only videos."
            )
    else:
        signal_text = (
            "There are not enough observations under this lens "
            "to summarize geographic reach reliably."
        )

    st.markdown(
        f"""
        <div class="signal">
            <div class="eyebrow">
                Geographic signal
            </div>
            <div style="
                font-size:1.35rem;
                margin-top:.45rem;
                line-height:1.45
            ">
                {signal_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # MARKET COMPOSITION HEATMAP
    # ---------------------------------------------------------
    st.markdown(
        "### The content mix changes by market"
    )

    heat = comp.copy()
    heat["share_pct"] = 100 * heat["share"]

    if category == "All categories":
        # Show the most common categories across the
        # selected time window.
        topcats = (
            categorized_base["category_name"]
            .value_counts()
            .head(8)
            .index
        )

        heat_display = heat[
            heat["category_name"].isin(topcats)
        ].copy()

    else:
        # Show only the selected category, BUT its share
        # is still calculated against each market's entire
        # categorized trending board.
        heat_display = heat[
            heat["category_name"].astype(str) == category
        ].copy()

    pivot = (
        heat_display
        .pivot(
            index="category_name",
            columns="region_code",
            values="share_pct",
        )
        .fillna(0)
    )

    pivot = pivot.reindex(
        columns=["US", "CA", "GB", "IN"],
        fill_value=0,
    )

    if not pivot.empty:
        fig = px.imshow(
            pivot,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale=[
                [0, BG],
                [1, GOLD],
            ],
            labels={
                "x": "Market",
                "y": "Category",
                "color": "Share %",
            },
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Market: %{x}<br>"
                "Share of market's categorized "
                "trending observations: %{z:.1f}%"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            xaxis_title=None,
            yaxis_title=None,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:
        st.info(
            "No category-composition observations are available "
            "under this lens."
        )

    # ---------------------------------------------------------
    # Bottom charts
    # ---------------------------------------------------------
    a, b = st.columns(2)

    # ---------------------------------------------------------
    # Reach distribution
    # ---------------------------------------------------------
    with a:
        st.markdown(
            "### Most videos appear in only one market"
        )

        rc = (
            reach.value_counts()
            .reindex(
                [1, 2, 3, 4],
                fill_value=0,
            )
            .rename_axis("markets")
            .reset_index(name="videos")
        )

        rc["market_label"] = rc["markets"].map(
            {
                1: "1 market",
                2: "2 markets",
                3: "3 markets",
                4: "4 markets",
            }
        )

        fig = px.bar(
            rc,
            x="market_label",
            y="videos",
            text="videos",
        )

        fig.update_traces(
            marker_color=GOLD,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Unique videos: %{y:,}"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=330,
            margin=dict(
                l=10,
                r=20,
                t=20,
                b=10,
            ),
            showlegend=False,
            xaxis_title=None,
            yaxis_title="Unique videos",
        )

        fig.update_yaxes(
            rangemode="tozero"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # Local vs cross-region velocity
    # ---------------------------------------------------------
    with b:
        st.markdown(
            "### Local vs cross-region velocity"
        )

        vd = (
            med.rename("median_velocity")
            .reset_index()
            .dropna()
        )

        # Keep a predictable order.
        vd["order"] = vd["reach_type"].map(
            {
                "Local only": 0,
                "Cross-region": 1,
            }
        )

        vd = vd.sort_values("order")

        vd["velocity_label"] = vd[
            "median_velocity"
        ].apply(
            lambda v: (
                f"{v / 1000:.1f}k/hr"
                if abs(v) >= 1000
                else f"{v:,.0f}/hr"
            )
        )

        if not vd.empty:
            max_velocity = vd[
                "median_velocity"
            ].max()

            fig = px.bar(
                vd,
                x="median_velocity",
                y="reach_type",
                orientation="h",
                text="velocity_label",
            )

            fig.update_traces(
                marker_color=GOLD,
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Median observed view velocity: "
                    "%{x:,.0f} views/hour"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                height=330,
                margin=dict(
                    l=10,
                    r=90,
                    t=20,
                    b=10,
                ),
                showlegend=False,
                yaxis_title=None,
            )

            fig.update_xaxes(
                title=(
                    "Median observed view velocity "
                    "(views/hour)"
                ),
                range=[
                    0,
                    max_velocity * 1.25
                    if max_velocity > 0
                    else 1,
                ],
            )

            fig.update_yaxes(
                categoryorder="array",
                categoryarray=[
                    "Local only",
                    "Cross-region",
                ],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "No view-velocity observations are available "
                "under this lens."
            )

        st.caption(
            "Snapshot-based median observed velocity. "
            "The comparison is descriptive and does not imply causation."
        )

    # ---------------------------------------------------------
    # So What
    # ---------------------------------------------------------
    if pd.notna(cross_share) and pd.notna(lift):
        so_what_text = (
            f"Most trending videos remain geographically concentrated: "
            f"<b>{cross_share:.0%}</b> appear in at least two collected "
            f"markets. But videos that do cross markets show "
            f"<b>{lift:.1f}×</b> the median observed view velocity of "
            f"local-only videos. The next question is: "
            f"<b>can we predict whether that attention survives tomorrow?</b>"
        )

    elif pd.notna(cross_share):
        so_what_text = (
            f"Trending remains strongly market-specific: "
            f"<b>{cross_share:.0%}</b> of unique videos under this lens "
            f"appear in at least two collected markets. "
            f"The next question is: "
            f"<b>can we predict whether attention survives tomorrow?</b>"
        )

    else:
        so_what_text = (
            "Geographic reach varies across the collected markets. "
            "The next question is: "
            "<b>can we predict whether attention survives tomorrow?</b>"
        )

    st.markdown(
        f"""
        <div class="so-what">
            <div class="gold-kicker">
                SO WHAT
            </div>
            <div style="
                font-size:1.12rem;
                margin-top:.5rem;
                line-height:1.55
            ">
                {so_what_text}
            </div>
            <div style="
                color:{GOLD_SOFT};
                margin-top:.9rem;
                font-weight:600
            ">
                Continue → 06 · Prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Methodology
    # ---------------------------------------------------------
    with st.expander("How this is calculated"):
        st.markdown(
            """
**Geographic reach.** A unique video is **local-only** when it appears
in exactly one of the four collected markets (CA, GB, IN or US) during
the selected time window. A **cross-region** video appears in at least
two collected markets. “All 4 markets” means the same video was
observed trending in CA, GB, IN and US during that window.

**Market content share.** Heatmap percentages use each market's full
categorized trending observations as the denominator. For example, if
Music is selected, the Music value for CA is Music observations in CA
divided by all categorized trending observations in CA — selecting
Music does not redefine Music as 100% of the market.

**Most locally distinct mix.** Each market's full category-share
distribution is compared with the overall collected category mix for
the selected time window. The displayed market has the largest
distributional difference. This is descriptive of the four collected
markets only.

**View velocity.** Velocity is snapshot-based observed view growth per
hour in the collected history. The chart compares median velocity for
local-only and cross-region observations. The velocity ratio is an
association in this dataset and should not be interpreted as a causal
effect.

**Coverage.** Results describe the collected YouTube trending snapshots
for CA, GB, IN and US and should not be interpreted as all YouTube
activity worldwide.
            """
        )

elif chapter == "06 · Prediction":
    st.markdown('<div class="gold-kicker">06 · PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Can we predict whether attention survives tomorrow?</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">A leakage-safe Random Forest estimates whether each video-region state will still be trending in the same market on the next calendar day.</div>', unsafe_allow_html=True)

    rf_metrics = {"ROC-AUC": 0.871263, "F1": 0.742539, "Precision": 0.794118, "Recall": 0.697253}
    for col, (label, value) in zip(st.columns(4), rf_metrics.items()):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="signal"><div class="eyebrow">MODEL OUTLOOK</div><div style="font-size:1.35rem;margin-top:.45rem;line-height:1.45">The Random Forest reached <b>0.871 ROC-AUC</b> on a future held-out period, outperforming logistic regression and the majority baseline. The model predicts persistence — it does not explain why a video remains trending.</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, .95])
    with left:
        st.markdown("### The Random Forest separates persistence best")
        st.caption("ROC-AUC on the chronological held-out test period. Higher is better; 0.50 is chance-level ranking performance.")
        comparison = pd.DataFrame({"Model": ["Majority baseline", "Logistic regression", "Random Forest"], "ROC-AUC": [0.500000, 0.811990, 0.871263]})
        fig = px.bar(comparison, x="ROC-AUC", y="Model", orientation="h", text="ROC-AUC")
        fig.update_traces(marker_color=["#555861", "#8B8E98", GOLD], texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig.add_vline(x=.5, line_dash="dash", line_color="#555861", line_width=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=340, margin=dict(l=10,r=70,t=15,b=10), showlegend=False, font=dict(color="#C7C8CD"))
        fig.update_xaxes(title="Held-out ROC-AUC", range=[0,1], gridcolor="#202227")
        fig.update_yaxes(title=None, categoryorder="array", categoryarray=["Majority baseline","Logistic regression","Random Forest"])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### What the model knows at prediction time")
        st.caption("These are model inputs, not claimed feature-importance rankings.")
        feature_groups = [
            ("Rank & momentum", "Current rank, previous rank and rank movement"),
            ("Observed persistence", "Observations, hours and days trending so far"),
            ("Audience growth", "Views, view growth and view / like / comment velocity"),
            ("Engagement", "Engagement rate and engagement movement"),
            ("Channel context", "Subscriber, channel-view and channel-video counts"),
            ("Current reach", "Markets trending today and cross-region status"),
            ("Context", "YouTube category and market"),
        ]
        for name, description in feature_groups:
            st.markdown(f'<div style="border-bottom:1px solid {BORDER};padding:.58rem 0"><b>{name}</b><div class="subtle" style="font-size:.86rem;margin-top:.15rem">{description}</div></div>', unsafe_allow_html=True)

    st.markdown("### Current persistence watch")
    st.caption("Latest model-scored video-market states. Probability is the saved Random Forest estimate that the same video remains trending in the same market on the next calendar day.")

    try:
        from ai.build_fact_package import add_prediction_results
        with st.spinner("Scoring the latest video-market states…"):
            scored, prediction_date = add_prediction_results(raw, intel["latest_video_state"])

        prob_col = "still_trending_tomorrow_probability"
        pred_col = "still_trending_tomorrow_prediction"
        scored[prob_col] = pd.to_numeric(scored.get(prob_col), errors="coerce")
        scored[pred_col] = pd.to_numeric(scored.get(pred_col), errors="coerce")
        scored["trending_rank"] = pd.to_numeric(scored.get("trending_rank"), errors="coerce")
        scored_ok = scored.dropna(subset=[prob_col]).copy()

        if scored_ok.empty:
            st.info("The saved model loaded, but no latest states had all features required for scoring.")
        else:
            scored_ok = scored_ok.sort_values([prob_col, "trending_rank"], ascending=[False, True])
            likely_share = (scored_ok[pred_col] == 1).mean() if scored_ok[pred_col].notna().any() else np.nan
            median_probability = scored_ok[prob_col].median()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Prediction date", pd.to_datetime(prediction_date).date().isoformat())
            with c2: st.metric("States scored", f"{len(scored_ok):,}")
            with c3: st.metric("Median persistence probability", f"{median_probability:.0%}")
            if pd.notna(likely_share):
                st.caption(f"At the model's default decision threshold, {likely_share:.0%} of scored latest states are classified as still trending tomorrow.")

            def render_prediction_watch(rows, probability_class):
                for _, row in rows.iterrows():
                    c_rank, c_thumb, c_title, c_prob = st.columns([.55, 1.2, 5.2, 1.55], vertical_alignment="center")
                    with c_rank:
                        rank = row.get("trending_rank")
                        st.markdown(f'<div class="rank-number">#{int(rank) if pd.notna(rank) else "—"}</div>', unsafe_allow_html=True)
                    with c_thumb:
                        thumb = youtube_thumbnail(safe_text(row.get("id"), ""))
                        if thumb:
                            st.image(thumb, use_container_width=True)
                    with c_title:
                        st.markdown(f'<div class="video-title">{safe_text(row.get("video_title"), "Untitled video")}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="video-meta">{safe_text(row.get("channel_name"))} · {safe_text(row.get("category_name"))} · {safe_text(row.get("region_code"), "—")}</div>', unsafe_allow_html=True)
                    with c_prob:
                        probability = float(row[prob_col])
                        st.markdown(f'<span class="{probability_class}">{probability:.0%}</span>', unsafe_allow_html=True)
                        st.caption("persistence probability")
                    st.divider()

            st.markdown("#### Highest persistence signals")
            st.caption("The six latest video-market states with the highest estimated probability of remaining trending tomorrow.")
            high_watch = scored_ok.head(6)
            render_prediction_watch(high_watch, "rise")

            st.markdown("#### Lowest persistence signals")
            st.caption("The four latest video-market states with the lowest estimated probability of remaining trending tomorrow — the model's strongest disappearance-risk signals.")
            low_watch = scored_ok.tail(4).sort_values([prob_col, "trending_rank"], ascending=[True, True])
            render_prediction_watch(low_watch, "fall")
    except Exception as exc:
        st.warning("Latest persistence scores are temporarily unavailable. Historical held-out model evaluation is still shown above.")
        with st.expander("Prediction technical details"):
            st.exception(exc)

    st.markdown(f"""<div class="so-what"><div class="gold-kicker">SO WHAT</div><div style="font-size:1.12rem;margin-top:.5rem;line-height:1.55">The model can rank current video-market states by their estimated chance of surviving into tomorrow, while the held-out test shows that this signal generalizes better than simpler baselines over the observed test period. The next step is to combine the project's calculated evidence into a <b>grounded executive brief.</b></div><div style="color:{GOLD_SOFT};margin-top:.9rem;font-weight:600">Continue → 07 · AI Brief</div></div>""", unsafe_allow_html=True)

    with st.expander("How this is calculated"):
        st.markdown("""
**Prediction target.** The target is whether the same video is observed trending in the same region on the next calendar day. If that region was not collected the next day, the outcome is unknown and is excluded from supervised training rather than labeled as a failure.

**Prediction timing.** Features are constructed from information available at or before the current end-of-day video-region state. Final lifecycle outcomes such as `last_seen` or completed lifetime are not model inputs.

**Temporal validation.** The verified Phase 8 run trained on older observations (Sep 1–29), purged Sep 30 because its next-day target touches the test period, and evaluated on the future Oct 1–7 period. Random Forest held-out performance was ROC-AUC 0.871, F1 0.743, precision 0.794 and recall 0.697. Logistic regression ROC-AUC was 0.812; the majority baseline was 0.500.

**Model inputs.** The saved Random Forest pipeline uses 18 numeric features plus category and region. Numeric inputs cover rank/momentum, persistence observed so far, audience growth and velocity, engagement, channel context and current cross-region reach.

**Current watch.** The persistence watch loads the saved Random Forest artifact and reuses the same Phase 8 feature engineering used during model development. Probabilities are predictive scores, not guarantees or causal effects.

**Limitations.** Results are based on a limited historical window and four collected markets (CA, GB, IN and US). Performance can change with audience behavior, category mix, missing enrichment or data drift. Repeated observations of videos can also create correlation within the dataset. No statistical or causal claim should be inferred from a high prediction probability.
        """)


elif chapter == "07 · AI Brief":
    st.markdown('<div class="gold-kicker">07 · AI BRIEF</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">What should we know about trending attention right now?</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">A grounded executive interpretation of evidence already calculated by the trend intelligence system.</div>', unsafe_allow_html=True)

    brief_type = st.selectbox(
        "Brief type",
        [
            "Overall brief",
            "Momentum brief",
            "Prediction brief",
            "Regional brief",
        ],
        key="ai_brief_type",
    )

    st.markdown(
        f'''<div class="signal"><div class="eyebrow">AI SIGNAL</div>
        <div style="font-size:1.25rem;margin-top:.45rem;line-height:1.5">
        Gemini synthesizes <b>Python-calculated evidence</b>. It does not calculate dashboard metrics, retrieve outside facts, or replace the underlying data.
        </div></div>''',
        unsafe_allow_html=True,
    )

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("GOOGLE_API_KEY is not configured in this environment. The analytics pages still work; generation requires the key.")

    if st.button("Generate grounded brief", type="primary", key="generate_grounded_brief"):
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Configure GOOGLE_API_KEY before generating the AI brief.")
        else:
            try:
                with st.spinner("Building the grounded fact package and generating the brief…"):
                    from ai.gemini_interpreter import ask_grounded_gemini
                    facts = load_ai_fact_package()
                    lens_focus = {
                        "Overall brief": "Prioritize the overall story across momentum, lifecycle, content, regions and persistence predictions. Connect the layers instead of listing them.",
                        "Momentum brief": "Prioritize current observed momentum: rank changes, view velocity, engagement movement and the strongest current movers. Do not invent a time-change claim.",
                        "Prediction brief": "Prioritize persistence-model estimates and the current evidence around the video-market states being scored. Predictions are estimates, never observed future outcomes.",
                        "Regional brief": "Prioritize CA, GB, IN and US differences plus cross-region versus region-specific behavior. Do not speculate about causes.",
                    }

                    question = f'''
Create a scan-friendly trend-intelligence brief for this lens: {brief_type}.

LENS FOCUS:
{lens_focus[brief_type]}

Use EXACTLY these three markdown sections and no others.

## Executive brief
Write only 2–3 sentences. Give the overall analytical story. Synthesize instead of dumping metrics.

## What changed
Give at most 3 bullets. Each bullet must describe an observed movement, difference or contrast supported by the facts. Start each bullet with a short bold label. Use only the minimum supporting number needed.

## What to watch
Give at most 3 bullets. Each bullet must identify a concrete signal worth monitoring next and explain why briefly. Predictions must be labeled as model estimates, not future outcomes.

READABILITY RULES:
- Do not repeat the same fact across sections.
- Do not expose Python/internal field names. Say "cross-market" instead of "cross_region" and "single-market" instead of "region_specific".
- Never output internal video IDs or database-style identifiers.
- Convert probabilities such as 0.582 to 58%.
- Format rank movement as whole positions.
- Format large velocities readably, e.g. 107.7K views/hour.
- Do not print raw engagement-rate movement decimals such as -0.0382. Say "engagement rate declined" unless a human-friendly percentage-point statement is directly justified by the supplied metric.
- Prefer natural language over variable names.
- Each bullet should contain one main conclusion and only the evidence needed to support it. Keep each bullet to at most 2 short sentences.
- No long paragraphs or raw metric dumps.
- Use only the FACT PACKAGE and no outside knowledge.
- Do not claim causation.
- Do not say that a prediction "suggests continued performance", "indicates it will persist", or otherwise turn a probability into an outcome. State the model estimate neutrally.
- Never use the words "significant", "significantly", "statistically significant", or related statistical-significance wording.
'''
                    brief = ask_grounded_gemini(question, facts)
                    st.session_state["ai_brief_text"] = brief
                    st.session_state["ai_brief_facts"] = facts
                    st.session_state["ai_brief_lens"] = brief_type
            except Exception as exc:
                st.error("The grounded AI brief could not be generated.")
                with st.expander("AI technical details"):
                    st.exception(exc)

    brief = st.session_state.get("ai_brief_text")
    facts = st.session_state.get("ai_brief_facts")

    if brief and facts:
        left, right = st.columns([1.65, 1], gap="large")
        with left:
            st.markdown("### Grounded executive brief")
            generated_lens = st.session_state.get("ai_brief_lens", brief_type)
            st.caption(f"Lens · {generated_lens} · generated only from the structured evidence package shown alongside it.")
            st.markdown(brief)

        with right:
            st.markdown("### Fact package")
            st.caption("The evidence supplied to Gemini — not facts generated by Gemini.")
            metadata = facts.get("metadata", {})
            rising = facts.get("rising_videos", []) or []
            categories = facts.get("category_trends", []) or []
            regions = facts.get("regional_trends", []) or []
            cross_region = facts.get("global_vs_regional", []) or []
            prediction_date = safe_text(metadata.get("prediction_date"), "—")
            region_names = metadata.get("regions", []) or []
            observations = metadata.get("historical_observations")

            rank_moves = [pd.to_numeric(x.get("rank_movement"), errors="coerce") for x in rising]
            rank_moves = [x for x in rank_moves if pd.notna(x)]
            strongest_move = max(rank_moves) if rank_moves else np.nan
            life = [pd.to_numeric(x.get("days_trending"), errors="coerce") for x in rising]
            life = [x for x in life if pd.notna(x)]
            median_life = float(pd.Series(life).median()) if life else np.nan
            probs = [pd.to_numeric(x.get("still_trending_tomorrow_probability"), errors="coerce") for x in rising]
            probs = [x for x in probs if pd.notna(x)]
            high_prob = max(probs) if probs else np.nan
            lead_category = safe_text(categories[0].get("category_name"), "—") if categories else "—"
            region_text = " · ".join(map(str, region_names)) if region_names else "—"

            fact_cards = [
                ("PULSE", f"{len(rising):,} current rising states supplied", f"Prediction date · {prediction_date}"),
                ("MOMENTUM", f"+{strongest_move:.0f} positions" if pd.notna(strongest_move) else "—", "Strongest rank move in supplied current evidence"),
                ("LIFECYCLE", f"{median_life:.1f} days" if pd.notna(median_life) else "—", "Median observed duration among supplied rising states"),
                ("CONTENT", lead_category, "Leading category in the supplied category evidence"),
                ("REGIONS", region_text, f"{len(regions):,} regional evidence rows · {len(cross_region):,} reach-comparison rows"),
                ("PREDICTION", f"{high_prob:.0%}" if pd.notna(high_prob) else "—", "Highest persistence estimate among supplied rising states"),
            ]
            for label, value, note in fact_cards:
                st.markdown(
                    f'''<div style="background:{PANEL};border:1px solid {BORDER};border-radius:12px;padding:1rem;margin-bottom:.65rem">
                    <div class="eyebrow">{label}</div><div style="font-size:1.05rem;font-weight:650;margin-top:.35rem">{value}</div>
                    <div class="subtle" style="font-size:.80rem;margin-top:.2rem">{note}</div></div>''',
                    unsafe_allow_html=True,
                )
            if observations is not None:
                try:
                    st.caption(f"{int(observations):,} historical observations support this fact package.")
                except (TypeError, ValueError):
                    st.caption(f"{observations} historical observations support this fact package.")
    else:
        st.markdown(
            f'''<div style="background:{PANEL};border:1px solid {BORDER};border-radius:14px;padding:1.7rem;margin-top:1rem">
            <div class="eyebrow">GROUNDED INTELLIGENCE</div><div style="font-size:1.22rem;font-weight:650;margin-top:.5rem">Generate the brief when you are ready.</div>
            <div class="subtle" style="margin-top:.45rem;line-height:1.55">The system will assemble structured evidence from the analytics and prediction layers, then ask Gemini to interpret only those supplied facts.</div></div>''',
            unsafe_allow_html=True,
        )

    st.markdown("### Grounding guardrails")
    for col, (title, body) in zip(st.columns(3), [
        ("No external facts", "No outside knowledge about videos, creators, events or markets is used to write the brief."),
        ("No unsupported statistics", "Gemini interprets supplied calculations and is explicitly told not to invent metrics or statistical significance."),
        ("Metrics remain source-of-truth", "Observed data, calculated analytics and model predictions stay separate from AI interpretation."),
    ]):
        with col:
            st.markdown(
                f'''<div style="background:{PANEL};border:1px solid {BORDER};border-radius:12px;padding:1rem;min-height:132px">
                <b>{title}</b><div class="subtle" style="font-size:.86rem;margin-top:.4rem;line-height:1.45">{body}</div></div>''',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'''<div class="so-what"><div class="gold-kicker">SO WHAT</div>
        <div style="font-size:1.12rem;margin-top:.5rem;line-height:1.55">The AI layer turns the project's calculated evidence into a readable decision narrative without replacing the underlying analytics. The final question is whether the data and pipeline supporting that story are healthy enough to trust.</div>
        <div style="color:{GOLD_SOFT};margin-top:.9rem;font-weight:600">Continue → 08 · Data Health</div></div>''',
        unsafe_allow_html=True,
    )

    with st.expander("How this is calculated"):
        st.markdown('''
**Fact package.** Python builds a structured evidence package from the historical analytics and saved Random Forest persistence model. Gemini does not calculate those metrics.

**Grounded generation.** Gemini receives the structured fact package plus strict rules requiring it to use only supplied evidence, distinguish predictions from observations, avoid causal claims and avoid unsupported statistical-significance language.

**Prediction language.** Persistence probability is a model estimate of whether the same video remains trending in the same collected market on the next calendar day. It is not an observed future outcome.

**Scope.** Geographic statements refer only to the four markets collected by this pipeline: CA, GB, IN and US.
        ''')

elif chapter == "08 · Data Health":
    st.markdown('<div class="gold-kicker">08 · DATA HEALTH</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Can we trust the data behind the story?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Validate collection coverage, enrichment, pipeline runs and the lifecycle records behind the dashboard.</div>',
        unsafe_allow_html=True,
    )

    q = raw.copy()
    q["fetch_timestamp"] = pd.to_datetime(q["fetch_timestamp"], utc=True, errors="coerce")
    q["fetch_date"] = pd.to_datetime(q["fetch_date"], errors="coerce").dt.normalize()

    region_days = (
        q.dropna(subset=["fetch_date", "region_code"])
        .groupby("fetch_date")["region_code"]
        .agg(lambda x: set(x.dropna().astype(str)))
    )
    complete_dates = [d for d, regs in region_days.items() if EXPECTED_REGIONS.issubset(regs)]
    latest_complete = max(complete_dates) if complete_dates else q["fetch_date"].max()
    latest_regions = region_days.get(latest_complete, set()) if pd.notna(latest_complete) else set()
    region_coverage = len(EXPECTED_REGIONS.intersection(latest_regions))

    channel_fields = [
        "channel_name",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count",
    ]
    present_channel_fields = [c for c in channel_fields if c in q.columns]
    enrichment_pct = (
        100 * q[present_channel_fields].notna().all(axis=1).mean()
        if present_channel_fields and len(q)
        else np.nan
    )
    category_pct = (
        100 * q["category_name"].notna().mean()
        if "category_name" in q.columns and len(q)
        else np.nan
    )
    duplicate_count = int(
        q.duplicated(subset=["id", "region_code", "fetch_timestamp"]).sum()
    )

    # Phase 7 snapshot rule: 100 expected records per market snapshot.
    snapshot_counts = (
        q.dropna(subset=["region_code", "fetch_timestamp"])
        .groupby(["region_code", "fetch_timestamp"])
        .size()
        .reset_index(name="records_received")
    )
    snapshot_counts["completion_pct"] = 100 * snapshot_counts["records_received"] / 100
    snapshot_counts["record_status"] = np.where(
        snapshot_counts["records_received"] >= 100,
        "Complete",
        "Below expected",
    )

    # Audit history is intentionally loaded only on page 08.
    try:
        audit = load_pipeline_audit()
        audit_error = None
    except Exception as exc:
        audit = pd.DataFrame()
        audit_error = str(exc)

    latest_audit_status = "Unavailable"
    freshness_hours = np.nan
    success_rate = np.nan

    if not audit.empty:
        a = audit.copy()
        for c in ["started_at", "ended_at"]:
            if c in a.columns:
                a[c] = pd.to_datetime(a[c], utc=True, errors="coerce")
        for c in ["records_fetched", "records_written", "retry_count"]:
            if c in a.columns:
                a[c] = pd.to_numeric(a[c], errors="coerce").fillna(0)

        if "ended_at" in a.columns:
            a = a.sort_values("ended_at")
            latest_run = a.iloc[-1]
            latest_audit_status = safe_text(latest_run.get("status"), "Unavailable")
            if pd.notna(latest_run.get("ended_at")):
                freshness_hours = (
                    pd.Timestamp.now(tz="UTC") - latest_run["ended_at"]
                ).total_seconds() / 3600

        if "status" in a.columns and len(a):
            success_rate = 100 * (a["status"] == "SUCCESS").mean()
    else:
        a = pd.DataFrame()

    pipeline_healthy = (
        latest_audit_status == "SUCCESS"
        and pd.notna(freshness_hours)
        and freshness_hours <= 9
        and region_coverage == len(EXPECTED_REGIONS)
        and duplicate_count == 0
    )
    pipeline_status = "Healthy" if pipeline_healthy else "Needs attention"

    latest_complete_text = (
        pd.to_datetime(latest_complete).strftime("%b %d")
        if pd.notna(latest_complete)
        else "—"
    )
    enrichment_text = f"{enrichment_pct:.0f}%" if pd.notna(enrichment_pct) else "—"

    for col, (label, value) in zip(st.columns(4), [
        ("Latest complete view", latest_complete_text),
        ("Region coverage", f"{region_coverage}/{len(EXPECTED_REGIONS)}"),
        ("Enrichment coverage", enrichment_text),
        ("Pipeline status", pipeline_status),
    ]):
        with col:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div></div>""",
                unsafe_allow_html=True,
            )

    if pipeline_healthy:
        health_message = (
            f"The latest audited run succeeded, the pipeline is within the 9-hour freshness threshold, "
            f"all {len(EXPECTED_REGIONS)} expected markets are represented in the latest complete view, "
            f"and no exact video-region-snapshot duplicates were detected."
        )
    else:
        issues = []
        if latest_audit_status != "SUCCESS":
            issues.append(f"latest audit status is {latest_audit_status}")
        if pd.isna(freshness_hours):
            issues.append("audit freshness is unavailable")
        elif freshness_hours > 9:
            issues.append(f"latest audited activity is {freshness_hours:.1f} hours old")
        if region_coverage < len(EXPECTED_REGIONS):
            issues.append(f"latest complete-view coverage is {region_coverage}/{len(EXPECTED_REGIONS)} markets")
        if duplicate_count:
            issues.append(f"{duplicate_count:,} exact duplicate observations were detected")
        health_message = "The system needs attention because " + "; ".join(issues) + "."

    st.markdown(
        f"""<div class="signal"><div class="eyebrow">DATA HEALTH SIGNAL</div>
        <div style="font-size:1.22rem;margin-top:.45rem;line-height:1.5">{health_message}</div></div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("### Channel enrichment by market")
        market_rows = []
        for reg in ["US", "CA", "GB", "IN"]:
            rq = q[q["region_code"] == reg]
            reg_enriched = (
                100 * rq[present_channel_fields].notna().all(axis=1).mean()
                if present_channel_fields and not rq.empty else np.nan
            )
            market_rows.append({
                "Market": reg,
                "Observations": len(rq),
                "Active days": rq["fetch_date"].nunique(),
                "Channel enrichment": reg_enriched,
            })

        market_df = pd.DataFrame(market_rows)
        fig = px.bar(
            market_df,
            x="Market",
            y="Channel enrichment",
            text="Channel enrichment",
            hover_data={"Observations": ":,", "Active days": True, "Channel enrichment": ":.1f"},
        )
        fig.update_traces(marker_color=GOLD, texttemplate="%{text:.0f}%", textposition="outside")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        fig.update_yaxes(title="Complete channel enrichment", range=[0, 105], ticksuffix="%")
        fig.update_xaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Complete channel enrichment requires channel name, subscriber count, channel view count and channel video count."
        )

    with right:
        st.markdown("### Pipeline path")
        for idx, (title, note) in enumerate([
            ("YouTube Data API", "Trending snapshots + video/channel metadata"),
            ("S3 snapshot history", "Timestamped raw history + run audits"),
            ("Historical intelligence", "Momentum, lifecycle, content and regional facts"),
            ("ML + grounded AI", "Persistence model + fact-bound interpretation"),
            ("Trend Intelligence", "Dashboard + drill-down evidence"),
        ], start=1):
            st.markdown(
                f"""<div style="background:{PANEL};border:1px solid {BORDER};border-radius:12px;padding:.8rem 1rem;margin-bottom:.5rem">
                <div style="display:flex;gap:.75rem;align-items:center">
                <div style="color:{GOLD};font-weight:800">{idx:02d}</div>
                <div><b>{title}</b><div class="subtle" style="font-size:.82rem;margin-top:.1rem">{note}</div></div>
                </div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### Recent pipeline checks")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audit success rate", f"{success_rate:.0f}%" if pd.notna(success_rate) else "—")
    c2.metric("Freshness", f"{freshness_hours:.1f} h" if pd.notna(freshness_hours) else "—")
    c3.metric("Exact duplicates", f"{duplicate_count:,}")
    c4.metric("Category coverage", f"{category_pct:.0f}%" if pd.notna(category_pct) else "—")

    if audit_error:
        st.caption(f"Audit history could not be loaded: {audit_error}")
    elif not a.empty:
        recent_cols = [
            c for c in [
                "ended_at", "status", "records_fetched", "records_written",
                "retry_count", "error_type"
            ] if c in a.columns
        ]
        recent = a.sort_values("ended_at", ascending=False)[recent_cols].head(8).copy()
        recent = recent.rename(columns={
            "ended_at": "Ended at",
            "status": "Status",
            "records_fetched": "Fetched",
            "records_written": "Written",
            "retry_count": "Retries",
            "error_type": "Error",
        })
        st.dataframe(recent, hide_index=True, use_container_width=True)
    else:
        st.info("No pipeline audit records were available.")

    with st.expander("Explore the underlying data — lifecycle & entry-rank drill-down"):
        st.caption(
            "One row represents one video × market lifecycle. Use this to inspect the exact entry-rank evidence behind 03 Lifecycle."
        )

        history = intel["video_history"].copy()
        history["fetch_date"] = pd.to_datetime(history["fetch_date"], errors="coerce").dt.normalize()
        history["trending_rank"] = pd.to_numeric(history["trending_rank"], errors="coerce")

        daily = (
            history.sort_values("fetch_timestamp")
            .drop_duplicates(["id", "region_code", "fetch_date"], keep="last")
            .copy()
        )

        lifecycle = (
            daily.sort_values(["id", "region_code", "fetch_date"])
            .groupby(["id", "region_code"], as_index=False)
            .agg(
                video_title=("video_title", "last"),
                channel_name=("channel_name", "last"),
                category_name=("category_name", "last"),
                first_observed=("fetch_date", "min"),
                last_observed=("fetch_date", "max"),
                observed_lifetime_days=("fetch_date", "nunique"),
                entry_rank=("trending_rank", "first"),
                latest_rank=("trending_rank", "last"),
                observations=("fetch_date", "size"),
            )
        )

        lifecycle["entry_rank_band"] = pd.cut(
            lifecycle["entry_rank"],
            bins=[0, 10, 50, 100, np.inf],
            labels=["Top 10", "Rank 11–50", "Rank 51–100", "101+"],
            include_lowest=True,
        )

        f1, f2, f3, f4 = st.columns(4)
        region_filter = f1.selectbox(
            "Market", ["All markets", "US", "CA", "GB", "IN"],
            key="health_explorer_region",
        )
        categories = sorted(lifecycle["category_name"].dropna().astype(str).unique())
        category_filter = f2.selectbox(
            "Category", ["All categories"] + categories,
            key="health_explorer_category",
        )
        band_filter = f3.selectbox(
            "Entry rank",
            ["All entry ranks", "Top 10", "Rank 11–50", "Rank 51–100", "101+"],
            key="health_explorer_rank_band",
        )
        title_query = f4.text_input(
            "Video title contains",
            key="health_explorer_title",
            placeholder="Search title",
        )

        filtered = lifecycle.copy()
        if region_filter != "All markets":
            filtered = filtered[filtered["region_code"] == region_filter]
        if category_filter != "All categories":
            filtered = filtered[filtered["category_name"].astype(str) == category_filter]
        if band_filter != "All entry ranks":
            filtered = filtered[filtered["entry_rank_band"].astype(str) == band_filter]
        if title_query.strip():
            filtered = filtered[
                filtered["video_title"].fillna("").str.contains(
                    title_query.strip(), case=False, regex=False
                )
            ]

        display = filtered[
    [
        "video_title",
        "region_code",
        "entry_rank",
        "entry_rank_band",
        "first_observed",
        "last_observed",
        "observed_lifetime_days",
        "latest_rank",
        "category_name",
        "channel_name",
        "observations",
    ]
    ].copy()

        display = display.rename(columns={
        "video_title": "Video",
        "region_code": "Market",
        "entry_rank": "Entry rank",
        "entry_rank_band": "Entry rank band",
        "first_observed": "First observed",
        "last_observed": "Last observed",
        "observed_lifetime_days": "Observed lifetime",
        "latest_rank": "Latest rank",
        "category_name": "Category",
        "channel_name": "Channel",
        "observations": "Observed days",
    })

        st.caption(f"{len(display):,} video-market lifecycles match these filters.")
        st.dataframe(
            display.sort_values(
                ["Observed lifetime", "Entry rank"],
                ascending=[False, True],
            ).head(500),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download filtered lifecycle CSV",
            data=display.to_csv(index=False),
            file_name="youtube_lifecycle_entry_rank_drilldown.csv",
            mime="text/csv",
            key="download_lifecycle_explorer",
        )

    st.markdown("### System status")
    status_cols = st.columns(4)
    status_items = [
        ("Collection", f"{region_coverage}/{len(EXPECTED_REGIONS)} markets", region_coverage == len(EXPECTED_REGIONS)),
        ("Enrichment", enrichment_text, pd.notna(enrichment_pct) and enrichment_pct >= 95),
        ("Audit", latest_audit_status, latest_audit_status == "SUCCESS"),
        ("Freshness", f"{freshness_hours:.1f} h" if pd.notna(freshness_hours) else "Unavailable", pd.notna(freshness_hours) and freshness_hours <= 9),
    ]
    for col, (label, value, ok) in zip(status_cols, status_items):
        with col:
            state = "Healthy" if ok else "Check"
            st.markdown(
                f"""<div style="background:{PANEL};border:1px solid {BORDER};border-radius:12px;padding:1rem">
                <div class="eyebrow">{label}</div>
                <div style="font-size:1.08rem;font-weight:650;margin-top:.35rem">{value}</div>
                <div class="subtle" style="font-size:.80rem;margin-top:.2rem">{state}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("How this is calculated"):
        st.markdown("""
**Snapshot completeness.** Phase 7 expects 100 records per market snapshot and flags snapshots below that count.

**Region completeness.** A complete view contains all four expected markets: CA, GB, IN and US.

**Enrichment coverage.** Channel enrichment is complete only when channel name, subscriber count, channel view count and channel video count are all present.

**Pipeline freshness.** The audit layer treats the pipeline as fresh when the latest successful run is no more than 9 hours old. Audit history also tracks success/failure, retries and fetched-versus-written record counts.

**Lifecycle explorer.** One lifecycle is one video × market. Entry rank is the first observed trending rank for that lifecycle. Observed lifetime counts distinct collected calendar days; it does not assume the video remained trending on unobserved days.
        """)

    st.markdown(
        f"""<div class="so-what"><div class="gold-kicker">SYSTEM COMPLETE</div>
        <div style="font-size:1.12rem;margin-top:.5rem;line-height:1.55">
        Collection health and lifecycle evidence are inspectable beneath the analytical story instead of being hidden behind the presentation layer.
        </div></div>""",
        unsafe_allow_html=True,
    )


st.divider()
st.caption("Portfolio project · YouTube Data API snapshots · CA, GB, IN and US · Not affiliated with YouTube")
