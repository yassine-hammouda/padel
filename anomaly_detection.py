import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_tournament_anomaly():
    df_raw     = pd.read_csv("data/clean_top_tournaments.csv")
    df_results = pd.read_csv("data/anomaly_results_tournaments.csv")
    return df_raw, df_results


@st.cache_data
def load_matches_anomaly():
    df = pd.read_csv("data/clean_matches.csv")
    return df


# ── Matches anomaly computation ───────────────────────────────────────────────

def compute_matches_anomaly(df):
    from sklearn.preprocessing import LabelEncoder
    df2 = df.copy()

    # Features
    df2['category_enc'] = LabelEncoder().fit_transform(df2['category'])
    df2['source_enc']   = LabelEncoder().fit_transform(df2['competition_source'])
    df2['court_enc']    = LabelEncoder().fit_transform(df2['court'].fillna('Unknown'))
    df2['played_at']    = pd.to_datetime(df2['played_at'], errors='coerce')
    df2['month']        = df2['played_at'].dt.month.fillna(0).astype(int)
    df2['dur_filled']   = df2['duration_minutes'].fillna(df2['duration_minutes'].median())
    df2['has_duration'] = df2['duration_minutes'].notna().astype(int)

    FEATURES = ['category_enc', 'round', 'index', 'source_enc',
                'court_enc', 'month', 'dur_filled', 'has_duration']
    X = df2[FEATURES]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(contamination=0.08, random_state=42)
    df2['iso_label'] = np.where(iso.fit_predict(X_scaled) == -1, 'Anomaly', 'Normal')
    df2['iso_score'] = iso.decision_function(X_scaled)

    # LOF
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.08)
    df2['lof_pred']  = lof.fit_predict(X_scaled)
    df2['lof_score'] = lof.negative_outlier_factor_
    df2['lof_label'] = np.where(df2['lof_pred'] == -1, 'Anomaly', 'Normal')

    # Consensus
    df2['consensus'] = df2.apply(
        lambda r: 'Both: Anomaly' if r['iso_label'] == 'Anomaly' and r['lof_label'] == 'Anomaly'
        else ('Both: Normal' if r['iso_label'] == 'Normal' and r['lof_label'] == 'Normal'
              else 'Models Disagree'), axis=1
    )
    return df2


# ── Render Tournaments anomaly ────────────────────────────────────────────────

def render_tournaments_anomaly():
    st.markdown("## 🏟️ Anomaly Detection — Tournaments")
    st.markdown(
        "Detecting unusual tournaments using **Isolation Forest** vs **Local Outlier Factor (LOF)**. "
        "Results computed in `Nb5_AnomalyDetection.ipynb`."
    )

    try:
        df_raw, df = load_tournament_anomaly()
    except FileNotFoundError:
        st.error("Results file not found. Please run `Nb5_AnomalyDetection.ipynb` first.")
        return

    # KPIs
    st.markdown("### 📊 Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tournaments",          len(df))
    c2.metric("Isolation Forest Anomalies", int((df["iso_label"] == "Anomaly").sum()))
    c3.metric("LOF Anomalies",              int((df["lof_label"] == "Anomaly").sum()))
    c4.metric("Both Models Agree",          int((df["consensus"] == "Both: Anomaly").sum()))
    st.markdown("---")

    # Scatter plots
    st.markdown("### 🗺️ Revenue vs Attendance — Anomaly Map")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(
            df, x="stadium_attendance", y="ticket_revenue_eur",
            color="iso_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            hover_data=["tournament_name", "location", "year"],
            size="total_viewers_millions", title="Isolation Forest",
            labels={"stadium_attendance": "Attendance", "ticket_revenue_eur": "Revenue (€)", "iso_label": ""}
        )
        fig1.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.scatter(
            df, x="stadium_attendance", y="ticket_revenue_eur",
            color="lof_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            hover_data=["tournament_name", "location", "year"],
            size="total_viewers_millions", title="Local Outlier Factor",
            labels={"stadium_attendance": "Attendance", "ticket_revenue_eur": "Revenue (€)", "lof_label": ""}
        )
        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Consensus
    st.markdown("### 🤝 Model Consensus")
    fig3 = px.scatter(
        df, x="stadium_attendance", y="ticket_revenue_eur",
        color="consensus",
        color_discrete_map={"Both: Normal": "#2EC878", "Both: Anomaly": "#E74C3C", "Models Disagree": "#F39C12"},
        hover_data=["tournament_name", "location", "year"],
        size="total_viewers_millions", title="Where both models agree",
        labels={"stadium_attendance": "Attendance", "ticket_revenue_eur": "Revenue (€)"}
    )
    fig3.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # Score distributions
    st.markdown("### 📉 Anomaly Score Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig4 = px.histogram(df, x="iso_score", color="iso_label",
                            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
                            title="Isolation Forest — Decision Score",
                            labels={"iso_score": "Score (lower = more anomalous)"}, nbins=15)
        fig4.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        fig5 = px.histogram(df, x="lof_score", color="lof_label",
                            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
                            title="LOF — Negative Outlier Factor",
                            labels={"lof_score": "Score (lower = more anomalous)"}, nbins=15)
        fig5.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig5, use_container_width=True)

    # Model comparison table
    st.markdown("### ⚖️ Model Comparison")
    iso_anom  = int((df["iso_label"] == "Anomaly").sum())
    lof_anom  = int((df["lof_label"] == "Anomaly").sum())
    agreement = round((df["iso_label"] == df["lof_label"]).mean() * 100, 1)
    comparison = pd.DataFrame({
        "Metric": ["Anomalies Detected", "Normal Points", "Anomaly Rate (%)",
                   "Avg Score (anomalies)", "Avg Score (normal)"],
        "Isolation Forest": [
            iso_anom, len(df) - iso_anom, f"{iso_anom/len(df)*100:.1f}%",
            f"{df[df['iso_label']=='Anomaly']['iso_score'].mean():.3f}",
            f"{df[df['iso_label']=='Normal']['iso_score'].mean():.3f}",
        ],
        "Local Outlier Factor": [
            lof_anom, len(df) - lof_anom, f"{lof_anom/len(df)*100:.1f}%",
            f"{df[df['lof_label']=='Anomaly']['lof_score'].mean():.3f}",
            f"{df[df['lof_label']=='Normal']['lof_score'].mean():.3f}",
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    consensus_count = int((df["consensus"] == "Both: Anomaly").sum())
    st.info(f"**Model Agreement Rate: {agreement}%** — Both models flagged the same **{consensus_count} tournaments** as anomalies.")

    # Anomaly tables
    st.markdown("### 🚨 Detected Anomalous Tournaments")
    display_cols = ["tournament_name", "location", "year", "ticket_revenue_eur",
                    "stadium_attendance", "total_viewers_millions", "prize_money_eur", "occupancy_rate"]
    tab1, tab2, tab3 = st.tabs(["Isolation Forest", "LOF", "Both Models"])
    with tab1:
        st.dataframe(df[df["iso_label"] == "Anomaly"][display_cols].sort_values("ticket_revenue_eur", ascending=False),
                     use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(df[df["lof_label"] == "Anomaly"][display_cols].sort_values("ticket_revenue_eur", ascending=False),
                     use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(df[df["consensus"] == "Both: Anomaly"][display_cols].sort_values("ticket_revenue_eur", ascending=False),
                     use_container_width=True, hide_index=True)

    # Business insights
    st.markdown("### 💡 Business Insights")
    st.markdown("""
- **High revenue, low attendance** → Premium/VIP pricing — investigate ticket strategy
- **High attendance, low revenue** → Underpriced tickets — revenue optimization opportunity
- **Low viewers despite full stadium** → Poor broadcast deal — media rights need renegotiating
- **Both models agree = Anomaly** → High-confidence outlier — prioritize for business review
- **Models disagree** → Borderline case — needs domain expert review
    """)


# ── Render Matches anomaly ────────────────────────────────────────────────────

def render_matches_anomaly():
    st.markdown("## ⚔️ Anomaly Detection — Matches")
    st.markdown(
        "Detecting unusual matches using **Isolation Forest** vs **Local Outlier Factor (LOF)**. "
        "Computed live from `clean_matches.csv`."
    )

    try:
        df_raw = load_matches_anomaly()
    except FileNotFoundError:
        st.error("Match data not found. Check `data/clean_matches.csv`.")
        return

    df = compute_matches_anomaly(df_raw)

    # KPIs
    st.markdown("### 📊 Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Matches",              f"{len(df):,}")
    c2.metric("IF Anomalies",               int((df["iso_label"] == "Anomaly").sum()))
    c3.metric("LOF Anomalies",              int((df["lof_label"] == "Anomaly").sum()))
    c4.metric("Both Models Agree",          int((df["consensus"] == "Both: Anomaly").sum()))
    c5.metric("Missing Duration",           f"{df['duration_minutes'].isna().sum()} (49%)")
    st.markdown("---")

    # Scatter: Duration vs Round
    st.markdown("### 🗺️ Duration vs Round — Anomaly Map")
    df_plot = df.copy()
    df_plot['dur_display'] = df_plot['duration_minutes'].fillna(0)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(
            df_plot, x="round", y="dur_display",
            color="iso_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            hover_data=["name", "category", "round_name", "court"],
            size_max=12, opacity=0.7,
            title="Isolation Forest — Duration vs Round",
            labels={"round": "Round Number", "dur_display": "Duration (min)", "iso_label": ""}
        )
        fig1.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df_plot, x="round", y="dur_display",
            color="lof_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            hover_data=["name", "category", "round_name", "court"],
            size_max=12, opacity=0.7,
            title="LOF — Duration vs Round",
            labels={"round": "Round Number", "dur_display": "Duration (min)", "lof_label": ""}
        )
        fig2.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: Category breakdown
    st.markdown("### 📊 Anomalies by Category & Circuit")
    col1, col2 = st.columns(2)
    with col1:
        cat_counts = df[df['iso_label'] == 'Anomaly']['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig3 = px.bar(cat_counts, x='Category', y='Count',
                      color='Category', color_discrete_sequence=["#E74C3C", "#F39C12"],
                      title="IF Anomalies by Category",
                      template="plotly_dark")
        fig3.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        src_counts = df[df['iso_label'] == 'Anomaly']['competition_source'].value_counts().reset_index()
        src_counts.columns = ['Circuit', 'Count']
        fig4 = px.bar(src_counts, x='Circuit', y='Count',
                      color='Circuit', color_discrete_sequence=["#3B9EE8", "#9B5DE5", "#F4A259"],
                      title="IF Anomalies by Circuit",
                      template="plotly_dark")
        fig4.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Consensus map
    st.markdown("### 🤝 Model Consensus")
    fig5 = px.scatter(
        df_plot, x="round", y="dur_display",
        color="consensus",
        color_discrete_map={
            "Both: Normal":    "#2EC878",
            "Both: Anomaly":   "#E74C3C",
            "Models Disagree": "#F39C12"
        },
        hover_data=["name", "category", "round_name", "competition_source"],
        opacity=0.7,
        title="Where both models agree on match anomalies",
        labels={"round": "Round", "dur_display": "Duration (min)"}
    )
    fig5.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig5, use_container_width=True)

    # Score distributions
    st.markdown("### 📉 Anomaly Score Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig6 = px.histogram(df, x="iso_score", color="iso_label",
                            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
                            title="Isolation Forest — Decision Score",
                            labels={"iso_score": "Score (lower = more anomalous)"}, nbins=20)
        fig6.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig6, use_container_width=True)
    with col2:
        fig7 = px.histogram(df, x="lof_score", color="lof_label",
                            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
                            title="LOF — Negative Outlier Factor",
                            labels={"lof_score": "Score (lower = more anomalous)"}, nbins=20)
        fig7.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig7, use_container_width=True)

    # Model comparison
    st.markdown("### ⚖️ Model Comparison")
    iso_anom  = int((df["iso_label"] == "Anomaly").sum())
    lof_anom  = int((df["lof_label"] == "Anomaly").sum())
    agreement = round((df["iso_label"] == df["lof_label"]).mean() * 100, 1)
    comparison = pd.DataFrame({
        "Metric": ["Anomalies Detected", "Normal Matches", "Anomaly Rate (%)",
                   "Avg Score (anomalies)", "Avg Score (normal)"],
        "Isolation Forest": [
            iso_anom, len(df) - iso_anom, f"{iso_anom/len(df)*100:.1f}%",
            f"{df[df['iso_label']=='Anomaly']['iso_score'].mean():.3f}",
            f"{df[df['iso_label']=='Normal']['iso_score'].mean():.3f}",
        ],
        "Local Outlier Factor": [
            lof_anom, len(df) - lof_anom, f"{lof_anom/len(df)*100:.1f}%",
            f"{df[df['lof_label']=='Anomaly']['lof_score'].mean():.3f}",
            f"{df[df['lof_label']=='Normal']['lof_score'].mean():.3f}",
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    consensus_count = int((df["consensus"] == "Both: Anomaly").sum())
    st.info(f"**Model Agreement Rate: {agreement}%** — Both models flagged **{consensus_count} matches** as anomalies.")

    # Anomaly tables


    # Business insights
    st.markdown("### 💡 Business Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**🕐 Duration Anomalies:**
- Very short matches (<60 min) → Possible walkover or retirement
- Very long matches (>180 min) → Super tiebreak or weather delays
- 49% missing duration → Major data quality issue to address

**🏆 Round Anomalies:**
- Finals with index=1 → Unexpected seeding results
- Early rounds with long duration → Potential upsets or retirements
        """)
    with col2:
        st.markdown("""
**📊 Circuit & Category Patterns:**
- FIP circuit dominates anomaly count → More variance in match conditions
- Women's matches with men's round structure → Data entry errors possible
- Off-season matches → Check competition calendar consistency

**🎯 Recommended Actions:**
- Flag missing duration matches for manual review
- Cross-check anomalous matches with official scorecards
- Set automated alerts for matches >180 min or <45 min
        """)


# ── Main render function (called from app.py) ─────────────────────────────────

def render_anomaly_detection():
    # Top selector
    st.markdown("### 🔍 Anomaly Detection")
    domain = st.radio(
        "Select domain:",
        ["🏟️ Tournaments", "⚔️ Matches"],
        horizontal=True,
        key="anomaly_domain_selector"
    )
    st.markdown("---")

    if domain == "🏟️ Tournaments":
        render_tournaments_anomaly()
    else:
        render_matches_anomaly()