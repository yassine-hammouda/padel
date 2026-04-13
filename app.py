# ============================================================
# PADEL ANALYTICS — ML STREAMLIT APP
# Players + Tournaments integrated
# ============================================================
from anomaly_detection import render_anomaly_detection
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import google.generativeai as genai

from groq import Groq
from matches_ml import (
    render_matches_classification,
    render_matches_regression,
    render_matches_clustering,
    render_matches_timeseries,
)
PADEL_CONTEXT = """You are a Padel Analytics expert assistant. You have access to a professional padel dataset:

PLAYERS & RANKINGS:
1. Players (1054): ranking, points, nationality, age, hand, side, height
2. Rankings (4111): Top men: Arturo Coello & Agustin Tapia (19800 pts), 96 countries
3. Player Titles: F. Belasteguin leads with 162 total titles
4. Brand Market Share: Bullpadel 28.5%, Adidas 22.3%, HEAD 19.7%, Siux 9.2%, Nox 6.8%
5. Rackets (730): avg price €179, range €6.9–€799.9, 28 vendors
6. Racket Performance: Bullpadel Hack 01 leads with 92.4% success rate at €389.9
7. Sponsorship: Estrella Damm €8.5M/year, 185% ROI
8. Player Contracts: Arturo Coello €450k with Bullpadel, avg contract €161,850
9. Social Media: Arturo Coello 2.8M Instagram followers, avg sponsorship value €533k/year
10. Shoes (78): avg €83.5, Asics leads with 26 products
11. FIP Titles: J. Luis Gonzalez & M. Cassetta lead with 13 titles each

TOURNAMENTS:
12. Tournaments (150): across 33 countries, Spain hosts the most (37)
13. Top Tournaments: Premier Padel Riyadh — 15.2M viewers, €3.2M revenue, 99% occupancy
14. Seasons: 6 seasons 2023–2026, best year 2025 with 40 tournaments

MATCHES (1,260 professional matches):
15. Categories: Men & Women, FIP & WPT circuits
16. Match Winner: ~50/50 split team_1 vs team_2 across all matches
17. Match Duration: avg ~105 min, range 60–180 min (624 matches missing duration)
18. Rounds: Finals, Semifinals, Quarterfinals, Round of 16, 32, etc.
19. Courts: Center Court = finals/semis, side courts = early rounds
20. Top players in matches: Tapia/Coello pair dominates men's finals (Season 5 & 6)
21. Season 5 & 6 have most matches, FIP circuit dominates the dataset
22. Longest matches tend to be men's finals (3 sets, super tiebreak)
23. Women's matches avg shorter by ~15 min vs men's matches

ML MODELS ON MATCHES:
24. Classification: XGBoost predicts match winner — 74% accuracy, ROC-AUC 0.79
25. Random Forest only achieves 55% accuracy on match winner prediction
26. Key predictors for winner: round type, category, court type, competition source
27. Regression: Predicting match duration — best model Ridge MAE=24 min, R²≈0 (hard to predict)
28. Clustering: KMeans k=2 separates men vs women matches clearly (Silhouette=0.216)
29. Time Series: ARIMA MAE=360 matches/month, Prophet MAE=549 (short series, limited accuracy)
30. Anomaly insight: 49% missing duration data is a major data quality flag

Answer concisely and professionally. Always cite specific numbers. Keep answers under 150 words unless a breakdown is explicitly requested. If asked about predictions, explain what the ML models found."""

SUGGESTIONS = [
    "Who are the top 5 ranked players?",
    "Which brand has the highest market share?",
    "Which player has the most titles?",
    "Which tournament had the best attendance?",
    "Which sponsor has the best ROI?",
    "Who has the most Instagram followers?",
    "How accurate is the match winner prediction?",
    "What's the average match duration?",
    "Which circuit has the most matches?",
    "What do the 2 match clusters represent?",
    "Why is match duration hard to predict?",
    "Who dominates men's finals in Season 5?",
]
# ── Paste your Groq key here ───────────────────────────────
GROQ_API_KEY = "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"


def render_chatbot():
    st.markdown("## 💬 NLP — Padel AI Chatbot")
    st.markdown("Ask anything about **players, tournaments, rackets, sponsorships, or match analytics**.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Topic filter ───────────────────────────────────────────
    topic = st.selectbox(
        "🔍 Filter quick questions by topic:",
        ["All", "🏆 Players & Rankings", "🏟️ Tournaments", "⚔️ Matches & ML"],
        label_visibility="visible"
    )

    topic_map = {
        "All": SUGGESTIONS,
        "🏆 Players & Rankings": [
            "Who are the top 5 ranked players?",
            "Which player has the most titles?",
            "Who has the most Instagram followers?",
            "Which brand has the highest market share?",
            "Which sponsor has the best ROI?",
        ],
        "🏟️ Tournaments": [
            "Which tournament had the best attendance?",
            "Which country hosts the most tournaments?",
            "What was the best season for tournaments?",
            "Which tournament has the highest revenue?",
        ],
        "⚔️ Matches & ML": [
            "How accurate is the match winner prediction?",
            "What's the average match duration?",
            "Which circuit has the most matches?",
            "What do the 2 match clusters represent?",
            "Why is match duration hard to predict?",
            "Who dominates men's finals in Season 5?",
        ],
    }

    filtered = topic_map[topic]

    st.markdown("**⚡ Quick questions:**")
    cols = st.columns(3)
    for i, s in enumerate(filtered):
        if cols[i % 3].button(s, key=f"chip_{topic}_{i}", use_container_width=True):
            st.session_state.pending_q = s

    st.divider()

    # ── Chat history ───────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Input ──────────────────────────────────────────────────
    prompt = st.chat_input("Ask about players, matches, ML models, tournaments...")
    if "pending_q" in st.session_state:
        prompt = st.session_state.pop("pending_q")

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    messages = [{"role": "system", "content": PADEL_CONTEXT}]
                    messages += st.session_state.chat_history
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        max_tokens=500,
                    )
                    reply = response.choices[0].message.content
                    st.write(reply)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply}
                    )
                except Exception as e:
                    st.error(f"API error: {str(e)}")

    # ── Footer stats ───────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", "4,111")
    c2.metric("Tournaments", "150")
    c3.metric("Matches", "1,260")
    c4.metric("Messages", len(st.session_state.chat_history))

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Padel Analytics ML",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        letter-spacing: 0.12em;
        color: #F0F0F0;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
        text-shadow: 0 2px 24px rgba(46,200,120,0.18);
    }
    .sub-header {
        font-size: 1rem;
        color: #8A9BB0;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .section-badge {
        display: inline-block;
        background: linear-gradient(90deg, #1A6640, #2EC878);
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 3px 14px;
        border-radius: 20px;
        margin-bottom: 0.7rem;
    }
    .winner-box {
        background: linear-gradient(135deg, #0d3d22 0%, #145c34 100%);
        border-left: 4px solid #2EC878;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: #d4f5e3;
    }
    .insight-box {
        background: #0f1c2e;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #b0c8e8;
    }
    .sidebar-section {
        font-size: 0.75rem;
        color: #2EC878;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 1.2rem 0 0.3rem 0;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 2px solid #1e3a5f; }
    .stTabs [data-baseweb="tab"] { height: 44px; font-size: 14px; font-weight: 600; color: #8A9BB0; }
    .stTabs [aria-selected="true"] { color: #2EC878 !important; }
    div[data-testid="metric-container"] {
        background: #101c2c;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.6rem 1rem;
    }
    hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_player_data():
    try:
        ranking = pd.read_csv('data/clean_dim_ranking.csv')
        players = pd.read_csv('data/clean_players.csv')
        seasons = pd.read_csv('data/clean_seasons.csv')
        return ranking, players, seasons
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_tournament_data():
    try:
        top_tours = pd.read_csv('data/clean_top_tournaments.csv')
        social    = pd.read_csv('data/clean_tournaments_social.csv')

        top_tours['match_key'] = top_tours['tournament_name'].str[:15].str.lower().str.strip()
        social['match_key']    = social['tournament_name'].str[:15].str.lower().str.strip()
        df = top_tours.merge(social, on='match_key', how='left', suffixes=('', '_social'))

        for col in ['instagram_reach_millions','youtube_views_millions',
                    'tiktok_videos_millions','engagement_rate_percent']:
            df[col] = df[col].fillna(0)

        df['prize_money_millions'] = df['prize_money_eur'] / 1_000_000
        df['revenue_per_seat']     = df['ticket_revenue_eur'] / df['stadium_capacity']
        df['digital_score'] = (
            df['instagram_reach_millions'] * 0.3 +
            df['youtube_views_millions']   * 0.3 +
            df['tiktok_videos_millions']   * 0.2 +
            df['engagement_rate_percent']  * 0.2
        )
        df['sold_out'] = (df['occupancy_rate'] >= 95).astype(int)

        df['physical_score'] = (
            (df['stadium_attendance'] / df['stadium_capacity']) * 40 +
            (df['stadium_capacity']   / df['stadium_capacity'].max()) * 30 +
            (df['prize_money_eur']    / df['prize_money_eur'].max()) * 20 +
            (df['ticket_revenue_eur'] / df['ticket_revenue_eur'].max()) * 10
        )
        df['dig_score'] = (
            (df['instagram_reach_millions'] / (df['instagram_reach_millions'].max() or 1)) * 25 +
            (df['youtube_views_millions']   / (df['youtube_views_millions'].max()   or 1)) * 25 +
            (df['tiktok_videos_millions']   / (df['tiktok_videos_millions'].max()   or 1)) * 20 +
            (df['total_viewers_millions']   / (df['total_viewers_millions'].max()   or 1)) * 20 +
            (df['engagement_rate_percent']  / (df['engagement_rate_percent'].max()  or 1)) * 10
        )
        df['broadcast_score'] = (
            (df['tv_channels_count']      / (df['tv_channels_count'].max()      or 1)) * 50 +
            (df['peak_viewers_thousands'] / (df['peak_viewers_thousands'].max() or 1)) * 50
        )
        for col in ['physical_score','dig_score','broadcast_score']:
            mn, mx = df[col].min(), df[col].max()
            df[col] = ((df[col] - mn) / (mx - mn + 1e-9)) * 100

        df = df.dropna(subset=['physical_score','dig_score','broadcast_score']).reset_index(drop=True)
        return df
    except FileNotFoundError:
        return None

ranking, players, seasons = load_player_data()
df_tour = load_tournament_data()
df_matches = pd.read_csv('data/clean_matches.csv')

# ============================================================
# SIDEBAR  — ONE single radio per section, session_state resolves active page
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.8rem;'
        'color:#2EC878;letter-spacing:0.1em;">🎾 PADEL ML</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown('<div class="sidebar-section">General</div>', unsafe_allow_html=True)
    gen_sel = st.radio("gen", ["🏠 Overview"],
                       label_visibility="collapsed", key="radio_gen")

    with st.expander("👤 Players", expanded=False):
        player_sel = st.radio("players", [
            "👤 Players — Classification",
            "👤 Players — Regression",
            "👤 Players — Clustering",
        ], label_visibility="collapsed", key="radio_players")

    with st.expander("🏟️ Tournaments", expanded=False):
        tour_sel = st.radio("tournaments", [
            "🏟️ Tournaments — Classification",
            "🏟️ Tournaments — Regression",
            "🏟️ Tournaments — Clustering",
            "🏟️ Tournaments — Time Series",
        ], label_visibility="collapsed", key="radio_tournaments")

    with st.expander("⚔️ Matches", expanded=False):
        match_sel = st.radio("matches", [
            "⚔️ Matches — Classification",
            "⚔️ Matches — Regression",
            "⚔️ Matches — Clustering",
            "⚔️ Matches — Time Series",
        ], label_visibility="collapsed", key="radio_matches")

    with st.expander("🤖 Advanced", expanded=False):
        nlp_sel = st.radio("nlp", [
            "💬 NLP — Chatbot",
            "🔍 Anomaly Detection",
        ], label_visibility="collapsed", key="radio_nlp")

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Dataset Info</div>', unsafe_allow_html=True)
    if ranking is not None:
        st.metric("Players",   f"{len(ranking):,}")
        st.metric("Countries", f"{ranking['country'].nunique()}")
    if df_tour is not None:
        st.metric("Tournaments", f"{len(df_tour)}")
    if df_matches is not None:
        st.metric("Matches", f"{len(df_matches):,}")

# ── session_state: track which group was touched last ──────
if "active_group" not in st.session_state:
    st.session_state.active_group  = "gen"
    st.session_state.prev_gen      = gen_sel
    st.session_state.prev_players  = player_sel
    st.session_state.prev_tour     = tour_sel
    st.session_state.prev_nlp      = nlp_sel   # NEW
    st.session_state.prev_matches  = match_sel


if gen_sel != st.session_state.prev_gen:
    st.session_state.active_group = "gen"
    st.session_state.prev_gen = gen_sel
elif player_sel != st.session_state.prev_players:
    st.session_state.active_group = "players"
    st.session_state.prev_players = player_sel
elif tour_sel != st.session_state.prev_tour:
    st.session_state.active_group = "tour"
    st.session_state.prev_tour = tour_sel
elif nlp_sel != st.session_state.prev_nlp:       # NEW
    st.session_state.active_group = "nlp"         # NEW
    st.session_state.prev_nlp = nlp_sel           # NEW
elif match_sel != st.session_state.prev_matches:
    st.session_state.active_group = "matches"
    st.session_state.prev_matches = match_sel

if st.session_state.active_group == "gen":
    active = gen_sel
elif st.session_state.active_group == "players":
    active = player_sel
elif st.session_state.active_group == "nlp":
    active = nlp_sel
elif st.session_state.active_group == "matches":
    active = match_sel
else:
    active = tour_sel

# ============================================================
# HELPERS
# ============================================================
PALETTE = ['#2EC878','#E84855','#3B9EE8','#F4A259','#9B5DE5','#F7C59F','#06D6A0','#EF476F']

def style_ax(ax):
    ax.set_facecolor('#0a1628')
    ax.figure.set_facecolor('#0a1628')
    ax.tick_params(colors='#8A9BB0')
    ax.xaxis.label.set_color('#8A9BB0')
    ax.yaxis.label.set_color('#8A9BB0')
    ax.title.set_color('#F0F0F0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.grid(True, alpha=0.15, color='#1e3a5f')

# ============================================================
# HEADER (always visible)
# ============================================================
st.markdown('<div class="main-header">🎾 Padel Analytics — ML Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Machine Learning Models · Professional Padel Circuit</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ============================================================
# ① OVERVIEW
# ============================================================
if active == "🏠 Overview":
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    if ranking is not None:
        col1.metric("Total Players",  f"{len(ranking):,}")
        col2.metric("Elite Players",  f"{(ranking['position'] <= 100).sum()}")
        col3.metric("Countries",      f"{ranking['country'].nunique()}")
        col4.metric("Avg Points",     f"{ranking['points'].mean():.0f}")
    col5.metric("Total Matches", f"{len(df_matches):,}")
    col6.metric("Match Winner Accuracy", "74%")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 Business Objective")
        st.info("""
        **Tournament & Player Analytics Platform**

        - 🏆 **Federations**: Identify elite talent early
        - 💰 **Sponsors**: Find best ROI player profiles
        - 🏟️ **Organizers**: Forecast tournament growth & revenue
        - 📊 **Analysts**: Segment player & tournament profiles
        - ⚔️ **Coaches**: Predict match outcomes & analyze patterns
        - 📡 **Broadcasters**: Forecast monthly match volume
        """)
        st.subheader("🤖 Models Built")
        st.success("""
        **👤 Players**
        → Classification: Elite prediction (RF vs XGBoost)
        → Regression: Points prediction (RF vs XGBoost)
        → Clustering: Strategic profiles (K-Means vs DBSCAN)

        **🏟️ Tournaments**
        → Classification: Sold-Out prediction (RF vs XGBoost)
        → Regression: Ticket revenue (Ridge vs RF vs XGBoost)
        → Clustering: Digital vs Physical segmentation
        → Time Series: Growth forecast (ARIMA vs Prophet)

        **⚔️ Matches**
        → Classification: Winner prediction (RF vs XGBoost) — 74% accuracy
        → Regression: Duration prediction (RF vs XGBoost)
        → Clustering: Match profiles (K-Means k=2)
        → Time Series: Monthly volume forecast (ARIMA vs Prophet)
        """)
    with c2:
        if ranking is not None:
            st.subheader("📈 Points Distribution")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(ranking['points'], bins=50, color='#2EC878', edgecolor='#0a1628', alpha=0.9)
            ax.set_yscale('log')
            style_ax(ax); ax.set_title('Player Points Distribution (log scale)')
            st.pyplot(fig); plt.close()

            st.subheader("🌍 Top 10 Countries")
            top_c = ranking['country'].value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.barh(top_c.index, top_c.values, color='#2EC878', alpha=0.85)
            style_ax(ax2); ax2.set_title('Players by Country')
            st.pyplot(fig2); plt.close()

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("⚔️ Matches by Category")
        cat_counts = df_matches['category'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.bar(cat_counts.index, cat_counts.values,
                color=['#2EC878','#E84855'], alpha=0.85, width=0.5)
        ax3.set_ylabel('Number of Matches')
        style_ax(ax3); ax3.set_title('Matches by Category (Men vs Women)')
        st.pyplot(fig3); plt.close()

    with col_right:
        st.subheader("📅 Matches per Season")
        season_counts = df_matches['season_year'].value_counts().sort_index()
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        ax4.plot(season_counts.index, season_counts.values,
                 'o-', color='#3B9EE8', linewidth=2, markersize=8)
        ax4.fill_between(season_counts.index, season_counts.values,
                         alpha=0.15, color='#3B9EE8')
        style_ax(ax4); ax4.set_title('Number of Matches per Season')
        ax4.set_xlabel('Season Year'); ax4.set_ylabel('Matches')
        st.pyplot(fig4); plt.close()
    st.markdown("---")
    col_left, col_right = st.columns(2)

# ============================================================
# ② CLASSIFICATION — PLAYERS
# ============================================================
elif active == "👤 Players — Classification":
    st.markdown('<span class="section-badge">👤 Players</span>', unsafe_allow_html=True)
    st.header("🏆 Elite Player Prediction")
    st.markdown("Predict whether a player will reach **Elite status (Top 100)**")

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Results", "💡 Insights"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            points_input = st.number_input("Current Points", 0, 20000, 500, 100)
            move_input   = st.slider("Ranking Movement", -200, 200, 0)
        with c2:
            gender_input      = st.selectbox("Gender", ["M", "F"])
            top_country_input = st.selectbox("From Top Padel Country?", ["Yes (ESP/ARG/BRA/POR/ITA)", "No"])

        if st.button("🎯 Predict Elite Status", type="primary"):
            le = LabelEncoder(); le.fit(["F","M"])
            gender_enc  = le.transform([gender_input])[0]
            top_country = 1 if "Yes" in top_country_input else 0
            points_log  = np.log1p(points_input)
            features    = np.array([[points_input, points_log, move_input, gender_enc, top_country]])
            try:
                rf_model    = joblib.load('models/random_forest_classifier.pkl')
                prediction  = rf_model.predict(features)[0]
                probability = rf_model.predict_proba(features)[0]
                st.markdown("---")
                if prediction == 1:
                    st.success("## ⭐ ELITE PLAYER PREDICTED!"); st.balloons()
                else:
                    st.warning("## 📊 NON-ELITE PLAYER")
                st.metric("Elite Probability", f"{probability[1]*100:.1f}%")
                fig, ax = plt.subplots(figsize=(8, 1.5))
                ax.barh([''], [probability[1]], color='#2EC878' if prediction==1 else '#E84855', height=0.5)
                ax.barh([''], [1-probability[1]], left=[probability[1]], color='#1e3a5f', height=0.5)
                ax.set_xlim(0,1); ax.axvline(0.5, color='grey', linestyle='--', alpha=0.5)
                style_ax(ax); ax.set_title(f'Elite Probability: {probability[1]*100:.1f}%')
                st.pyplot(fig); plt.close()
            except FileNotFoundError:
                st.error("Model not found. Run the Classification notebook first.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Random Forest")
            st.metric("Accuracy","100%"); st.metric("F1-Score","1.0000")
            st.metric("ROC-AUC","1.0000"); st.metric("CV Mean F1","0.9975 ± 0.0049")
        with c2:
            st.markdown("### XGBoost")
            st.metric("Accuracy","100%"); st.metric("F1-Score","1.0000")
            st.metric("ROC-AUC","1.0000"); st.metric("CV Mean F1","0.9902 ± 0.0091")
        st.markdown('<div class="winner-box">🏆 <b>Winner: Random Forest</b> — Both models achieve perfect test scores. RF wins on CV stability (lower std = more reliable in production).</div>', unsafe_allow_html=True)
        feats = ['points','points_log','move','gender_encoded','top_country']
        fig, ax = plt.subplots(figsize=(8,3))
        ax.barh(feats, [0.45,0.35,0.10,0.06,0.04], color='#2EC878', alpha=0.85)
        style_ax(ax); ax.set_xlabel('Importance'); ax.set_title('Random Forest — Feature Importance')
        st.pyplot(fig); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="insight-box"><b>Federations 🏛️</b><br>Points accumulation is the primary predictor. Track players with high positive move scores. Early identification saves development costs.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Sponsors 💰</b><br>Target Rising Stars before Elite status. Better ROI than signing established Elites. Use move score to spot emerging talent.</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="insight-box"><b>Organizers 🏟️</b><br>Elite players attract larger audiences. Mix Elite + Rising Stars for compelling events.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Key Finding 🔑</b><br>Points is the strongest predictor (45%). Gender and country matter less than performance. Ranking movement reveals momentum.</div>', unsafe_allow_html=True)

# ============================================================
# ③ REGRESSION — PLAYERS
# ============================================================
elif active == "👤 Players — Regression":
    st.markdown('<span class="section-badge">👤 Players</span>', unsafe_allow_html=True)
    st.header("📉 Player Points Prediction")
    st.markdown("Predict a player's **ranking points** based on their profile")

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Results", "💡 Insights"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            position_input = st.number_input("Current Ranking Position", 1, 3000, 500, 10)
            move_input     = st.slider("Ranking Movement", -500, 500, 0)
        with c2:
            gender_input      = st.selectbox("Gender", ["M","F"])
            top_country_input = st.selectbox("From Top Padel Country?", ["Yes (ESP/ARG/BRA/POR/ITA)","No"])

        if st.button("🎯 Predict Points", type="primary"):
            le = LabelEncoder(); le.fit(["F","M"])
            gender_enc  = le.transform([gender_input])[0]
            top_country = 1 if "Yes" in top_country_input else 0
            features    = np.array([[position_input, 1/position_input, move_input,
                                      max(move_input,0), abs(min(move_input,0)),
                                      gender_enc, top_country, 0.03]])
            try:
                rf_reg   = joblib.load('models/rf_regressor.pkl')
                pred_pts = np.expm1(rf_reg.predict(features)[0])
                tier     = "⭐ Elite" if pred_pts>1000 else "🏅 Professional" if pred_pts>200 else "🔼 Developing" if pred_pts>50 else "📈 Beginner Pro"
                est_pos  = max(1, int(3000*(1-pred_pts/20000)))
                st.markdown("---")
                st.success(f"## 🎯 Predicted Points: **{pred_pts:.0f}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Points", f"{pred_pts:.0f}")
                c2.metric("Player Tier", tier)
                c3.metric("Estimated Position", f"~{est_pos}")
            except FileNotFoundError:
                st.error("Model not found. Run the Regression notebook first.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Random Forest Regressor")
            st.metric("R² Score","~0.97"); st.metric("MAE","~45 points"); st.metric("RMSE","~180 points")
        with c2:
            st.markdown("### XGBoost Regressor")
            st.metric("R² Score","~0.96"); st.metric("MAE","~52 points"); st.metric("RMSE","~195 points")
        st.markdown('<div class="winner-box">🏆 <b>Winner: Random Forest Regressor</b> — Higher R² + Lower RMSE + Better CV stability.</div>', unsafe_allow_html=True)
        feats_reg = ['position','position_inverse','move','move_positive','move_negative','gender_enc','top_country','country_freq']
        fig, ax = plt.subplots(figsize=(8,3))
        ax.barh(feats_reg, [0.05,0.65,0.08,0.05,0.05,0.04,0.04,0.04], color='steelblue', alpha=0.85)
        style_ax(ax); ax.set_xlabel('Importance'); ax.set_title('RF — Feature Importance')
        st.pyplot(fig); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="insight-box"><b>Federations 🏛️</b><br>Forecast player point trajectory. Plan wildcard invitations. Spot declining players for development support.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Sponsors 💰</b><br>Predict future points before signing deals. Target players with high predicted growth.</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="insight-box"><b>Analysts 📊</b><br>Model explains 97% variance (R²=0.97). Position inverse is dominant predictor.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Key Insight 🔑</b><br>A player ranked 500 with +200 move gains points faster than one ranked 300 with −50 move.</div>', unsafe_allow_html=True)

# ============================================================
# ④ CLUSTERING — PLAYERS
# ============================================================
elif active == "👤 Players — Clustering":
    st.markdown('<span class="section-badge">👤 Players</span>', unsafe_allow_html=True)
    st.header("👥 Player Segmentation")
    st.markdown("Segment players into **strategic profiles** for targeted decisions")

    tab1, tab2, tab3 = st.tabs(["🗺️ Segments", "📊 Model Results", "💡 Insights"])

    with tab1:
        if ranking is not None:
            df_ml = ranking.copy()
            df_ml['gender_encoded'] = LabelEncoder().fit_transform(df_ml['gender'])
            df_ml['points_log']     = np.log1p(df_ml['points'])
            df_ml['elite_score']    = (df_ml['points'] / df_ml['points'].max()) * 100
            feat_cols = ['points_log','position','move','gender_encoded','elite_score']
            X_ps = StandardScaler().fit_transform(df_ml[feat_cols].fillna(0))
            try:
                labels_p = joblib.load('models/kmeans_clustering.pkl').predict(X_ps)
            except FileNotFoundError:
                labels_p = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(X_ps)
            df_ml['cluster'] = labels_p

            pca_p = PCA(n_components=2, random_state=42)
            X_pca_p = pca_p.fit_transform(X_ps)

            c1, c2 = st.columns(2)
            with c1:
                cs = df_ml['cluster'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(6,4))
                ax.pie(cs.values, labels=[f'Cluster {i}' for i in cs.index],
                       autopct='%1.1f%%', colors=PALETTE[:len(cs)])
                ax.set_facecolor('#0a1628'); fig.set_facecolor('#0a1628')
                ax.set_title('Player Distribution by Cluster', color='#F0F0F0')
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(X_pca_p[:,0], X_pca_p[:,1], c=labels_p, cmap='tab10', alpha=0.5, s=8)
                style_ax(ax)
                ax.set_xlabel(f'PC1 ({pca_p.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca_p.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title('PCA 2D — Player Segments')
                st.pyplot(fig); plt.close()

            st.subheader("📋 Cluster Profiles")
            st.dataframe(df_ml.groupby('cluster').agg(
                Count=('points','count'), Avg_Points=('points','mean'),
                Avg_Position=('position','mean'), Avg_Move=('move','mean')
            ).round(1), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### K-Means (K=8)")
            st.metric("Silhouette Score","0.5647"); st.metric("Davies-Bouldin","0.5881")
            st.metric("Noise Points","0"); st.metric("Interpretability","HIGH")
        with c2:
            st.markdown("### DBSCAN")
            st.metric("Silhouette Score","0.4244"); st.metric("Davies-Bouldin","0.6291")
            st.metric("Noise Points","105"); st.metric("Interpretability","MEDIUM")
        st.markdown('<div class="winner-box">🏆 <b>Winner: K-Means</b> — Higher Silhouette + Lower Davies-Bouldin + Zero noise points.</div>', unsafe_allow_html=True)

    with tab3:
        for seg, desc in {
            "⭐ Elite (Top 3)":      "10 players — Coello, Tapia, Galan level. Premium sponsorship targets.",
            "⭐ Elite Pro (Top 50)": "149 players — Tour regulars. Strong sponsorship ROI.",
            "🔼 Rising Stars":       "445 players — Mid-ranking, stable. Best ROI for sponsors NOW.",
            "🚀 Rockets":            "14 players — Low ranking but +587 move. Sign them before price rises!",
            "📉 Declining":          "9 players — High drop (−440 move). Don't renew contracts.",
            "📈 Developing":         "1900+ players — Building their career. Federation development focus.",
        }.items():
            st.markdown(f'<div class="insight-box"><b>{seg}</b><br>{desc}</div>', unsafe_allow_html=True)

# ============================================================
# ⑤ CLASSIFICATION — TOURNAMENTS
# ============================================================
elif active == "🏟️ Tournaments — Classification":
    st.markdown('<span class="section-badge">🏟️ Tournaments</span>', unsafe_allow_html=True)
    st.header("🎯 Tournament Sold-Out Prediction")
    st.markdown("Predict if a tournament will **sell out (≥95% occupancy)** before the event")

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Results", "💡 Insights"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            prize_m  = st.number_input("Prize Money (EUR millions)", 0.1, 2.0, 0.5, 0.05)
            viewers_m= st.number_input("Expected Total Viewers (millions)", 1.0, 20.0, 8.0, 0.5)
            tv_ch    = st.slider("Number of TV Channels", 1, 5, 1)
            capacity = st.number_input("Stadium Capacity", 5000, 25000, 12000, 500)
        with c2:
            peak_v   = st.number_input("Peak Viewers (thousands)", 500, 3000, 1500, 100)
            dig_sc   = st.number_input("Digital Score (0–100)", 0.0, 100.0, 40.0, 1.0)
            is_prem  = st.selectbox("Premium Location?", ["No","Yes (Barcelona/Madrid/Paris/Riyadh/Dubai)"])
            rev_seat = st.number_input("Revenue per Seat (EUR)", 50, 300, 120, 10)

        if st.button("🎯 Predict Sold-Out", type="primary"):
            ip  = 1 if "Yes" in is_prem else 0
            vpa = viewers_m / (capacity/1000) if capacity>0 else 0
            features = np.array([[prize_m, viewers_m, tv_ch, rev_seat, dig_sc, ip, vpa, capacity, peak_v]])
            try:
                rf_clf = joblib.load('models/rf_classifier_tournaments.pkl')
                pred   = rf_clf.predict(features)[0]
                proba  = rf_clf.predict_proba(features)[0]
                st.markdown("---")
                if pred==1: st.success("## ✅ SOLD OUT PREDICTED!"); st.balloons()
                else:       st.warning("## ❌ NOT SOLD OUT PREDICTED")
                st.metric("Sold-Out Probability", f"{proba[1]*100:.1f}%")
                fig, ax = plt.subplots(figsize=(8,1.5))
                ax.barh([''], [proba[1]], color='#2EC878' if pred==1 else '#E84855', height=0.5)
                ax.barh([''], [1-proba[1]], left=[proba[1]], color='#1e3a5f', height=0.5)
                ax.set_xlim(0,1); ax.axvline(0.5, color='grey', linestyle='--', alpha=0.5)
                style_ax(ax); ax.set_title(f'Sold-Out Probability: {proba[1]*100:.1f}%')
                st.pyplot(fig); plt.close()
            except FileNotFoundError:
                st.error("Model not found. Run tournaments_classification notebook first.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Random Forest")
            st.metric("LOO CV F1","see notebook"); st.metric("ROC-AUC","see notebook")
        with c2:
            st.markdown("### XGBoost")
            st.metric("LOO CV F1","see notebook"); st.metric("ROC-AUC","see notebook")
        st.info("Run `tournaments_classification.ipynb` and paste your results here.")
        feats_tc = ['total_viewers_M','prize_money_M','digital_score','revenue_per_seat',
                    'stadium_capacity','peak_viewers_K','tv_channels','viewers_per_attendee','is_premium']
        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(feats_tc, [0.22,0.18,0.15,0.13,0.11,0.09,0.06,0.04,0.02], color='#2EC878', alpha=0.85)
        style_ax(ax); ax.set_xlabel('Importance'); ax.set_title('RF — Feature Importance (illustrative)')
        st.pyplot(fig); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="insight-box"><b>If SOLD OUT predicted ✅</b><br>→ Open additional temporary seating<br>→ Launch premium ticket tier early<br>→ Activate waitlist system immediately</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Key Predictors 📊</b><br>1. Total Viewers → demand for seats<br>2. Prize Money → fan interest<br>3. Digital Score → viral = fast fill<br>4. Revenue/Seat → pricing strategy</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="insight-box"><b>If NOT Sold Out predicted ❌</b><br>→ Increase marketing budget 6 weeks before<br>→ Partner with digital influencers<br>→ Offer early bird discounts</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>For Sponsors 💰</b><br>→ Prioritise sold-out predicted tournaments<br>→ Guaranteed full stadium visibility<br>→ Higher broadcast engagement expected</div>', unsafe_allow_html=True)

# ============================================================
# ⑥ REGRESSION — TOURNAMENTS
# ============================================================
elif active == "🏟️ Tournaments — Regression":
    st.markdown('<span class="section-badge">🏟️ Tournaments</span>', unsafe_allow_html=True)
    st.header("💶 Ticket Revenue Prediction")
    st.markdown("Predict **ticket revenue (EUR)** before the tournament to budget accurately")

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Results", "💡 Insights"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            cap_in     = st.number_input("Stadium Capacity", 5000, 25000, 12000, 500)
            prize_in   = st.number_input("Prize Money (EUR millions)", 0.1, 2.0, 0.4, 0.05)
            viewers_in = st.number_input("Expected Total Viewers (millions)", 1.0, 20.0, 8.0, 0.5)
            peak_in    = st.number_input("Peak Viewers (thousands)", 500, 3000, 1400, 100)
        with c2:
            tv_in   = st.slider("Number of TV Channels", 1, 5, 1)
            dig_in  = st.number_input("Digital Score (0–100)", 0.0, 100.0, 35.0, 1.0)
            prem_in = st.selectbox("Premium Location?", ["No","Yes (Barcelona/Madrid/Paris/Riyadh/Dubai)"])
            occ_in  = st.slider("Expected Occupancy (%)", 50, 100, 90)

        if st.button("💶 Predict Revenue", type="primary"):
            ip_r  = 1 if "Yes" in prem_in else 0
            features_r = np.array([[cap_in, prize_in, viewers_in, peak_in, tv_in, dig_in, ip_r, occ_in/100]])
            try:
                rf_reg_t = joblib.load('models/rf_regressor_tournaments.pkl')
                pred_eur = np.expm1(rf_reg_t.predict(features_r)[0])
                tier_t   = "🌟 Premium" if pred_eur>2e6 else "🏅 Standard" if pred_eur>1.2e6 else "🌱 Developing"
                st.markdown("---")
                st.success(f"## 💶 Predicted Revenue: **€{pred_eur:,.0f}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Revenue", f"€{pred_eur/1e6:.2f}M")
                c2.metric("Revenue per Seat",  f"€{pred_eur/cap_in:.0f}")
                c3.metric("Revenue Tier", tier_t)
            except FileNotFoundError:
                st.error("Model not found. Run tournaments_Regression notebook first.")

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Ridge")
            st.metric("R²","0.9410"); st.metric("MAE","€107,539"); st.metric("MAPE","6.55%")
        with c2:
            st.markdown("### Random Forest")
            st.metric("R²","0.9786"); st.metric("MAE","€59,998"); st.metric("MAPE","3.42%")
        with c3:
            st.markdown("### XGBoost")
            st.metric("R²","0.9694"); st.metric("MAE","€71,973"); st.metric("MAPE","4.05%")
        st.markdown('<div class="winner-box">🏆 <b>Winner: Random Forest</b> — R²=0.9786 | MAE=€59,998 | MAPE=3.42% · Predicts revenue within ~3.4% on average.</div>', unsafe_allow_html=True)
        feats_tr = ['stadium_capacity','prize_money_M','total_viewers_M','peak_viewers_K',
                    'tv_channels','digital_score','is_premium','capacity_utilization']
        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(feats_tr, [0.28,0.22,0.18,0.12,0.08,0.06,0.04,0.02], color='steelblue', alpha=0.85)
        style_ax(ax); ax.set_xlabel('Importance'); ax.set_title('RF — Feature Importance')
        st.pyplot(fig); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="insight-box"><b>Budget Planning 📋</b><br>Run prediction 3 months before tournament.<br>Conservative budget = prediction − €60K.<br>Stretch target = prediction + €30K.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Pricing Strategy 💰</b><br>If predicted revenue ≪ avg → raise prize money or boost digital campaigns.<br>If predicted revenue ≫ avg → add premium VIP / courtside tiers.</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="insight-box"><b>Key Revenue Drivers 📊</b><br>1. Stadium Capacity → biggest lever<br>2. Total Viewers → audience drives demand<br>3. Prize Money → prestige attracts fans<br>4. Digital Score → buzz = box office</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Investor Reporting 📈</b><br>Use model output as revenue forecast in pitch deck.<br>Confidence interval: prediction ± €59,998.<br>Model explains 97.9% of revenue variance.</div>', unsafe_allow_html=True)

# ============================================================
# ⑦ CLUSTERING — TOURNAMENTS
# ============================================================
elif active == "🏟️ Tournaments — Clustering":
    st.markdown('<span class="section-badge">🏟️ Tournaments</span>', unsafe_allow_html=True)
    st.header("🗂️ Tournament Segmentation")
    st.markdown("Segment tournaments by **Digital vs Physical impact** to allocate marketing budgets")

    tab1, tab2, tab3 = st.tabs(["🗺️ Segments", "📊 Model Results", "💡 Insights"])

    with tab1:
        if df_tour is not None:
            X_ts = StandardScaler().fit_transform(df_tour[['physical_score','dig_score','broadcast_score']])
            try:
                labels_t = joblib.load('models/kmeans_tournaments.pkl').predict(X_ts)
            except FileNotFoundError:
                labels_t = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_ts)
            df_tour['cluster'] = labels_t

            pca_t   = PCA(n_components=2, random_state=42)
            X_tpca  = pca_t.fit_transform(X_ts)
            df_tour['pca1'] = X_tpca[:,0]; df_tour['pca2'] = X_tpca[:,1]

            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(7,5))
                for i, (cl, grp) in enumerate(df_tour.groupby('cluster')):
                    ax.scatter(grp['dig_score'], grp['physical_score'],
                               label=f'Cluster {cl}', color=PALETTE[i], s=100,
                               alpha=0.85, edgecolors='white', linewidth=1)
                    for _, row in grp.iterrows():
                        ax.annotate(row['tournament_name'][:14], (row['dig_score'], row['physical_score']),
                                    fontsize=6, color='#8A9BB0', xytext=(3,3), textcoords='offset points')
                ax.axhline(50, color='#1e3a5f', linestyle='--', alpha=0.6)
                ax.axvline(50, color='#1e3a5f', linestyle='--', alpha=0.6)
                style_ax(ax); ax.set_xlabel('Digital Score'); ax.set_ylabel('Physical Score')
                ax.set_title('Digital vs Physical Matrix')
                ax.legend(facecolor='#0a1628', labelcolor='#F0F0F0', fontsize=8)
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(7,5))
                for i, (cl, grp) in enumerate(df_tour.groupby('cluster')):
                    ax.scatter(grp['pca1'], grp['pca2'], label=f'Cluster {cl}',
                               color=PALETTE[i], s=100, alpha=0.85, edgecolors='white', linewidth=1)
                    for _, row in grp.iterrows():
                        ax.annotate(row['tournament_name'][:12], (row['pca1'], row['pca2']),
                                    fontsize=6, color='#8A9BB0', xytext=(3,3), textcoords='offset points')
                style_ax(ax)
                ax.set_xlabel(f'PC1 ({pca_t.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca_t.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title('PCA 2D Projection')
                ax.legend(facecolor='#0a1628', labelcolor='#F0F0F0', fontsize=8)
                st.pyplot(fig); plt.close()

            st.subheader("💶 Avg Ticket Revenue by Cluster")
            seg_rev = df_tour.groupby('cluster')['ticket_revenue_eur'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8,3))
            bars = ax.bar([f'Cluster {i}' for i in seg_rev.index],
                          seg_rev.values/1e6, color=[PALETTE[i] for i in seg_rev.index], alpha=0.85)
            for bar, val in zip(bars, seg_rev.values/1e6):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'€{val:.2f}M', ha='center', fontsize=10, fontweight='bold', color='#F0F0F0')
            style_ax(ax); ax.set_ylabel('Avg Revenue (EUR M)'); ax.set_title('Average Ticket Revenue by Cluster')
            st.pyplot(fig); plt.close()

            st.subheader("📋 Cluster Profiles")
            st.dataframe(df_tour.groupby('cluster').agg(
                N=('tournament_name','count'),
                Avg_Physical=('physical_score','mean'), Avg_Digital=('dig_score','mean'),
                Avg_Broadcast=('broadcast_score','mean'), Avg_Revenue=('ticket_revenue_eur','mean'),
                Avg_Occupancy=('occupancy_rate','mean')
            ).round(2), use_container_width=True)
        else:
            st.error("Tournament data not found. Check `data/` folder.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### K-Means"); st.metric("Interpretability","HIGH"); st.metric("Noise Points","0")
        with c2:
            st.markdown("### Hierarchical (Ward)"); st.metric("Interpretability","HIGH"); st.metric("Dendrogram","✅ available")
        st.markdown('<div class="winner-box">🏆 Best algorithm selected automatically by Silhouette score in the notebook. Validated on Silhouette, Davies-Bouldin, and Calinski-Harabasz.</div>', unsafe_allow_html=True)

    with tab3:
        for seg, desc in {
            "🌟 Blockbuster":    "High physical + high digital. Top-tier events (Riyadh, Barcelona). Strategy: premium sponsorship packages, expand capacity.",
            "📱 Digital Giants": "Viral online reach, moderate stadium. Monetise via streaming & brand deals. Convert digital fans to ticket buyers.",
            "🏟️ Stadium Kings":  "Packed venues, lower digital buzz. Invest in TikTok content. Amplify highlights with influencer partnerships.",
            "🌱 Emerging":       "Developing events, growth potential. Build audience before expanding venues. Free streaming + social challenges.",
        }.items():
            st.markdown(f'<div class="insight-box"><b>{seg}</b><br>{desc}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📊 Recommended Marketing Budget Allocation")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.pie([40,30,20,10],
               labels=['🌟 Blockbuster','📱 Digital Giants','🏟️ Stadium Kings','🌱 Emerging'],
               autopct='%1.0f%%', colors=PALETTE[:4], startangle=140, pctdistance=0.75)
        ax.set_facecolor('#0a1628'); fig.set_facecolor('#0a1628')
        ax.set_title('Marketing Budget Allocation', color='#F0F0F0')
        st.pyplot(fig); plt.close()

# ============================================================
# ⑧ TIME SERIES — TOURNAMENTS
# ============================================================
elif active == "🏟️ Tournaments — Time Series":
    st.markdown('<span class="section-badge">🏟️ Tournaments</span>', unsafe_allow_html=True)
    st.header("📈 Tournament Growth Forecasting")
    st.markdown("Forecast the number of padel tournaments per year **(2025–2027)**")

    tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📊 Model Results", "💡 Insights"])

    with tab1:
        historical = pd.DataFrame({
            'year': [2016,2017,2018,2019,2020,2021,2022],
            'tournaments_count': [4,5,6,8,3,7,9]
        })
        if seasons is not None:
            ts_data = pd.concat(
                [historical, seasons[['year','tournaments_count']].sort_values('year')],
                ignore_index=True
            ).sort_values('year').reset_index(drop=True)
        else:
            ts_data = historical

        c1, c2, c3 = st.columns(3)
        c1.metric("2025 Forecast","~40","tournaments")
        c2.metric("2026 Forecast","~48","tournaments")
        c3.metric("2027 Forecast","~55","tournaments")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(ts_data['year'], ts_data['tournaments_count'],
                'o-', color='#2EC878', linewidth=2, markersize=8, label='Historical')
        ax.plot([2025,2026,2027],[40,48,55],
                'o--', color='#E84855', linewidth=2, markersize=8, label='Forecast')
        ax.fill_between([2025,2026,2027],[35,42,48],[45,54,62],
                        alpha=0.15, color='#E84855', label='95% CI')
        ax.axvline(2024, color='grey', linestyle=':', alpha=0.7, label='Forecast Start')
        ax.axvline(2020, color='#F4A259', linestyle='--', alpha=0.5, label='COVID Impact')
        style_ax(ax)
        ax.legend(facecolor='#0a1628', labelcolor='#8A9BB0')
        ax.set_xlabel('Year'); ax.set_ylabel('Number of Tournaments')
        ax.set_title('Padel Tournament Growth Forecast (2016–2027)')
        st.pyplot(fig); plt.close()
        st.info("📊 Strong upward trend 2025–2027 | COVID 2020 dip — full recovery achieved | ~15–20% growth per year")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ARIMA(1,1,1)")
            st.metric("MAE","~3.2"); st.metric("RMSE","~4.1"); st.metric("MAPE","~12.3%")
        with c2:
            st.markdown("### Prophet")
            st.metric("MAE","~2.8"); st.metric("RMSE","~3.6"); st.metric("MAPE","~10.1%")
        st.markdown('<div class="winner-box">🏆 <b>Winner: Prophet</b> — Lower MAE, RMSE and MAPE. Better handles the COVID structural break. Provides uncertainty intervals for risk planning.</div>', unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="insight-box"><b>Federations 🏛️</b><br>Plan infrastructure for ~55 tournaments by 2027. Hire officials ahead of demand. Expand to Middle East and Asia.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>Sponsors 💰</b><br>More tournaments = more visibility. Lock in multi-year deals NOW. ROI improves as audience grows.</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="insight-box"><b>Organizers 🏟️</b><br>Book venues 2 years in advance. Digital streaming rights becoming more valuable. Prize money expected to rise with growth.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box"><b>SDG Alignment 🌍</b><br>ODD 8: Tournament growth drives economic activity. ODD 17: International partnerships growing. Digital engagement connects global audiences.</div>', unsafe_allow_html=True)
elif active == "💬 NLP — Chatbot":
    render_chatbot()
elif active == "🔍 Anomaly Detection":
    render_anomaly_detection()
elif active == "⚔️ Matches — Classification":
    render_matches_classification()
elif active == "⚔️ Matches — Regression":
    render_matches_regression()
elif active == "⚔️ Matches — Clustering":
    render_matches_clustering()
elif active == "⚔️ Matches — Time Series":
    render_matches_timeseries()
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8A9BB0;font-size:0.85rem;letter-spacing:0.06em;'>
    🎾 PADEL ANALYTICS ML DASHBOARD &nbsp;
</div>
""", unsafe_allow_html=True)