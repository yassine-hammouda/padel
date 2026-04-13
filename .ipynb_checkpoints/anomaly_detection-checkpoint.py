import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/clean_top_tournaments.csv")
    df_results = pd.read_csv("data/anomaly_results_tournaments.csv")
    return df_raw, df_results


def render_anomaly_detection():
    st.markdown("## 🔍 Anomaly Detection — Tournaments")
    st.markdown(
        "Detecting unusual tournaments using **Isolation Forest** vs **Local Outlier Factor (LOF)**. "
        "Results computed in `Nb5_AnomalyDetection.ipynb`."
    )

    try:
        df_raw, df = load_data()
    except FileNotFoundError:
        st.error(
            "Results file not found. Please run `Nb5_AnomalyDetection.ipynb` first "
            "to generate `data/anomaly_results_tournaments.csv`."
        )
        return

    # ── KPI cards ──────────────────────────────────────────────
    st.markdown("### 📊 Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tournaments", len(df))
    c2.metric("Isolation Forest Anomalies", int((df["iso_label"] == "Anomaly").sum()))
    c3.metric("LOF Anomalies",              int((df["lof_label"] == "Anomaly").sum()))
    c4.metric("Both Models Agree",          int((df["consensus"] == "Both: Anomaly").sum()))

    st.markdown("---")

    # ── Scatter plots ──────────────────────────────────────────
    st.markdown("### 🗺️ Revenue vs Attendance — Anomaly Map")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            df,
            x="stadium_attendance", y="ticket_revenue_eur",
            color="iso_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            hover_data=["tournament_name", "location", "year"],
            size="total_viewers_millions",
            title="Isolation Forest",
            labels={"stadium_attendance": "Attendance", "ticket_revenue_eur": "Revenue (€)", "iso_label": ""}
        )
        fig1.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x="stadium_attendance", y="ticket_revenue_eur",
            color="lof_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            hover_data=["tournament_name", "location", "year"],
            size="total_viewers_millions",
            title="Local Outlier Factor",
            labels={"stadium_attendance": "Attendance", "ticket_revenue_eur": "Revenue (€)", "lof_label": ""}
        )
        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Consensus map ──────────────────────────────────────────
    st.markdown("### 🤝 Model Consensus")
    fig3 = px.scatter(
        df,
        x="stadium_attendance", y="ticket_revenue_eur",
        color="consensus",
        color_discrete_map={
            "Both: Normal":    "#2EC878",
            "Both: Anomaly":   "#E74C3C",
            "Models Disagree": "#F39C12"
        },
        hover_data=["tournament_name", "location", "year"],
        size="total_viewers_millions",
        title="Where both models agree",
        labels={"stadium_attendance": "Attendance", "ticket_revenue_eur": "Revenue (€)"}
    )
    fig3.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Score distributions ────────────────────────────────────
    st.markdown("### 📉 Anomaly Score Distributions")
    col1, col2 = st.columns(2)

    with col1:
        fig4 = px.histogram(
            df, x="iso_score", color="iso_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            title="Isolation Forest — Decision Score",
            labels={"iso_score": "Score (lower = more anomalous)"},
            nbins=15
        )
        fig4.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        fig5 = px.histogram(
            df, x="lof_score", color="lof_label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E74C3C"},
            title="LOF — Negative Outlier Factor",
            labels={"lof_score": "Score (lower = more anomalous)"},
            nbins=15
        )
        fig5.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Model comparison table ─────────────────────────────────
    st.markdown("### ⚖️ Model Comparison")

    iso_anom  = int((df["iso_label"] == "Anomaly").sum())
    lof_anom  = int((df["lof_label"] == "Anomaly").sum())
    agreement = round((df["iso_label"] == df["lof_label"]).mean() * 100, 1)

    comparison = pd.DataFrame({
        "Metric": [
            "Anomalies Detected", "Normal Points", "Anomaly Rate (%)",
            "Avg Score (anomalies)", "Avg Score (normal)"
        ],
        "Isolation Forest": [
            iso_anom,
            len(df) - iso_anom,
            f"{iso_anom / len(df) * 100:.1f}%",
            f"{df[df['iso_label']=='Anomaly']['iso_score'].mean():.3f}",
            f"{df[df['iso_label']=='Normal']['iso_score'].mean():.3f}",
        ],
        "Local Outlier Factor": [
            lof_anom,
            len(df) - lof_anom,
            f"{lof_anom / len(df) * 100:.1f}%",
            f"{df[df['lof_label']=='Anomaly']['lof_score'].mean():.3f}",
            f"{df[df['lof_label']=='Normal']['lof_score'].mean():.3f}",
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    consensus_count = int((df["consensus"] == "Both: Anomaly").sum())
    st.info(
        f"**Model Agreement Rate: {agreement}%** — "
        f"Both models flagged the same **{consensus_count} tournaments** as anomalies."
    )

    # ── Anomaly tables ─────────────────────────────────────────
    st.markdown("### 🚨 Detected Anomalous Tournaments")

    display_cols = [
        "tournament_name", "location", "year",
        "ticket_revenue_eur", "stadium_attendance",
        "total_viewers_millions", "prize_money_eur", "occupancy_rate"
    ]

    tab1, tab2, tab3 = st.tabs(["Isolation Forest", "LOF", "Both Models"])

    with tab1:
        st.dataframe(
            df[df["iso_label"] == "Anomaly"][display_cols].sort_values(
                "ticket_revenue_eur", ascending=False),
            use_container_width=True, hide_index=True
        )
    with tab2:
        st.dataframe(
            df[df["lof_label"] == "Anomaly"][display_cols].sort_values(
                "ticket_revenue_eur", ascending=False),
            use_container_width=True, hide_index=True
        )
    with tab3:
        st.dataframe(
            df[df["consensus"] == "Both: Anomaly"][display_cols].sort_values(
                "ticket_revenue_eur", ascending=False),
            use_container_width=True, hide_index=True
        )

    # ── Business insights ──────────────────────────────────────
    st.markdown("### 💡 Business Insights")
    st.markdown("""
- **High revenue, low attendance** → Premium/VIP pricing — investigate ticket strategy
- **High attendance, low revenue** → Underpriced tickets — revenue optimization opportunity
- **Low viewers despite full stadium** → Poor broadcast deal — media rights need renegotiating
- **Both models agree = Anomaly** → High-confidence outlier — prioritize for business review
- **Models disagree** → Borderline case — needs domain expert review
    """)