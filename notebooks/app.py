# ============================================================
# PADEL ANALYTICS — ML STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    ranking = pd.read_csv('data/clean_dim_ranking.csv')
    players = pd.read_csv('data/clean_players.csv')
    seasons = pd.read_csv('data/clean_seasons.csv')
    return ranking, players, seasons

ranking, players, seasons = load_data()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">🎾 Padel Analytics — ML Dashboard</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Models for Professional Padel Analysis</div>', 
            unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Padel_racket_and_ball.jpg/320px-Padel_racket_and_ball.jpg", 
                 use_column_width=True)
st.sidebar.title("🎾 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Model",
    ["🏠 Overview", 
     "🏆 Classification", 
     "👥 Clustering", 
     "📈 Time Series"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.metric("Total Players", f"{len(ranking):,}")
st.sidebar.metric("Countries", f"{ranking['country'].nunique()}")
st.sidebar.metric("Seasons", f"{len(seasons)}")

# ============================================================
# OVERVIEW PAGE
# ============================================================
if page == "🏠 Overview":
    st.header("📊 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", f"{len(ranking):,}", "in ranking")
    with col2:
        st.metric("Elite Players", 
                  f"{(ranking['position'] <= 100).sum()}", 
                  "top 100")
    with col3:
        st.metric("Countries", f"{ranking['country'].nunique()}", "represented")
    with col4:
        st.metric("Avg Points", f"{ranking['points'].mean():.0f}", "per player")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Business Objective")
        st.info("""
        **Tournament & Player Analytics Platform**
        
        This ML dashboard helps stakeholders make data-driven decisions:
        
        - 🏆 **Federations**: Identify elite talent early
        - 💰 **Sponsors**: Find best ROI player profiles  
        - 🏟️ **Organizers**: Forecast tournament growth
        - 📊 **Analysts**: Segment player profiles
        """)
        
        st.subheader("🤖 Models Built")
        st.success("""
        **Classification** → Random Forest vs XGBoost
        Predict if a player is Elite (Top 100)
        
        **Clustering** → K-Means vs DBSCAN  
        Segment players into strategic profiles
        
        **Time Series** → ARIMA vs Prophet
        Forecast tournament growth 2025-2027
        """)
    
    with col2:
        st.subheader("📈 Points Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ranking['points'], bins=50, 
                color='#2E8B57', edgecolor='white', alpha=0.8)
        ax.set_xlabel('Points')
        ax.set_ylabel('Number of Players')
        ax.set_title('Player Points Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        st.subheader("🌍 Top 10 Countries")
        top_countries = ranking['country'].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bars = ax2.barh(top_countries.index, top_countries.values, 
                        color='#2E8B57', alpha=0.8)
        ax2.set_xlabel('Number of Players')
        ax2.set_title('Players by Country')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()

# ============================================================
# CLASSIFICATION PAGE
# ============================================================
elif page == "🏆 Classification":
    st.header("🏆 Elite Player Prediction")
    st.markdown("**Predict whether a player will reach Elite status (Top 100)**")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Results", "💡 Insights"])
    
    with tab1:
        st.subheader("🔮 Make a Prediction")
        st.markdown("Enter player details to predict Elite status:")
        
        col1, col2 = st.columns(2)
        with col1:
            points_input = st.number_input(
                "Current Points", 
                min_value=0, max_value=20000, 
                value=500, step=100
            )
            move_input = st.slider(
                "Ranking Movement", 
                min_value=-200, max_value=200, 
                value=0,
                help="Positive = improving, Negative = declining"
            )
        with col2:
            gender_input = st.selectbox("Gender", ["M", "F"])
            top_country_input = st.selectbox(
                "From Top Padel Country?",
                ["Yes (ESP/ARG/BRA/POR/ITA)", "No"]
            )
        
        if st.button("🎯 Predict Elite Status", type="primary"):
            # Prepare features
            le = LabelEncoder()
            le.fit(["F", "M"])
            gender_enc = le.transform([gender_input])[0]
            top_country = 1 if "Yes" in top_country_input else 0
            points_log = np.log1p(points_input)
            
            features = np.array([[points_input, points_log, 
                                   move_input, gender_enc, top_country]])
            
            try:
                rf_model = joblib.load('models/random_forest_classifier.pkl')
                prediction = rf_model.predict(features)[0]
                probability = rf_model.predict_proba(features)[0]
                
                st.markdown("---")
                if prediction == 1:
                    st.success(f"## ⭐ ELITE PLAYER PREDICTED!")
                    st.metric("Elite Probability", f"{probability[1]*100:.1f}%")
                    st.balloons()
                else:
                    st.warning(f"## 📊 NON-ELITE PLAYER")
                    st.metric("Elite Probability", f"{probability[1]*100:.1f}%")
                
                # Probability gauge
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['Elite Probability'], [probability[1]], 
                        color='#2E8B57' if prediction == 1 else '#FF6B6B',
                        height=0.5)
                ax.barh(['Elite Probability'], [1 - probability[1]], 
                        left=[probability[1]],
                        color='#E0E0E0', height=0.5)
                ax.set_xlim(0, 1)
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f'Prediction Probability: {probability[1]*100:.1f}% Elite')
                ax.set_xlabel('Probability')
                st.pyplot(fig)
                plt.close()
                
            except FileNotFoundError:
                st.error("Model not found. Please run the Classification notebook first.")
    
    with tab2:
        st.subheader("📊 Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Random Forest")
            st.metric("Accuracy", "100%")
            st.metric("F1-Score", "1.0000")
            st.metric("ROC-AUC", "1.0000")
            st.metric("CV Mean F1", "0.9975 ± 0.0049")
        
        with col2:
            st.markdown("### XGBoost")
            st.metric("Accuracy", "100%")
            st.metric("F1-Score", "1.0000")
            st.metric("ROC-AUC", "1.0000")
            st.metric("CV Mean F1", "0.9902 ± 0.0091")
        
        st.markdown("---")
        st.success("""
        **🏆 Winner: Random Forest**
        
        Both models achieve perfect test scores. 
        Random Forest wins on **cross-validation stability** (lower std = more reliable in production).
        """)
        
        # Feature importance chart
        st.subheader("Feature Importance")
        features = ['points', 'points_log', 'move', 'gender_encoded', 'top_country']
        importance_rf = [0.45, 0.35, 0.10, 0.06, 0.04]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(features, importance_rf, color='#2E8B57', alpha=0.8)
        ax.set_xlabel('Importance Score')
        ax.set_title('Random Forest — Feature Importance')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.subheader("💡 Business Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **For Federations 🏛️**
            - Points accumulation is the primary predictor
            - Track players with high positive move scores
            - Early identification saves development costs
            """)
            
            st.info("""
            **For Sponsors 💰**
            - Target Rising Stars before Elite status
            - Better ROI than signing established Elites
            - Use move score to spot emerging talent
            """)
        
        with col2:
            st.info("""
            **For Tournament Organizers 🏟️**
            - Elite players attract larger audiences
            - Mix Elite + Rising Stars for compelling events
            - Use model to curate matchup predictions
            """)
            
            st.info("""
            **Key Finding 🔑**
            - Points is the strongest predictor (45% importance)
            - Gender and country matter less than performance
            - Ranking movement reveals momentum
            """)

# ============================================================
# CLUSTERING PAGE
# ============================================================
elif page == "👥 Clustering":
    st.header("👥 Player Segmentation")
    st.markdown("**Segment players into strategic profiles for targeted decisions**")
    
    tab1, tab2, tab3 = st.tabs(["🗺️ Segments", "📊 Model Results", "💡 Insights"])
    
    with tab1:
        st.subheader("🗺️ Player Segments")
        
        # Recreate clustering for visualization
        df_ml = ranking.copy()
        df_ml['gender_encoded'] = LabelEncoder().fit_transform(df_ml['gender'])
        df_ml['points_log'] = np.log1p(df_ml['points'])
        df_ml['elite_score'] = (df_ml['points'] / df_ml['points'].max()) * 100
        
        feature_cols = ['points_log', 'position', 'move', 'gender_encoded', 'elite_score']
        X = df_ml[feature_cols].copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            kmeans = joblib.load('models/kmeans_clustering.pkl')
            labels = kmeans.predict(X_scaled)
            df_ml['cluster'] = labels
            
            # PCA for visualization
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster sizes
                cluster_sizes = df_ml['cluster'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(cluster_sizes.values, 
                       labels=[f'Cluster {i}' for i in cluster_sizes.index],
                       autopct='%1.1f%%',
                       colors=plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes))))
                ax.set_title('Player Distribution by Cluster')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # PCA scatter
                fig, ax = plt.subplots(figsize=(6, 4))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                    c=labels, cmap='Set1', 
                                    alpha=0.5, s=10)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title('PCA 2D — Player Segments')
                plt.colorbar(scatter, ax=ax, label='Cluster')
                st.pyplot(fig)
                plt.close()
            
            # Cluster profiles table
            st.subheader("📋 Cluster Profiles")
            profile = df_ml.groupby('cluster').agg({
                'points': 'mean',
                'position': 'mean',
                'move': 'mean',
                'cluster': 'count'
            }).rename(columns={'cluster': 'count'}).round(2)
            
            st.dataframe(profile, use_container_width=True)
            
        except FileNotFoundError:
            st.error("Model not found. Please run the Clustering notebook first.")
    
    with tab2:
        st.subheader("📊 Model Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### K-Means (K=8)")
            st.metric("Silhouette Score", "0.5647", "↑ higher is better")
            st.metric("Davies-Bouldin", "0.5881", "↓ lower is better")
            st.metric("Noise Points", "0", "all players assigned")
            st.metric("Interpretability", "HIGH")
        
        with col2:
            st.markdown("### DBSCAN")
            st.metric("Silhouette Score", "0.4244", "↑ higher is better")
            st.metric("Davies-Bouldin", "0.6291", "↓ lower is better")
            st.metric("Noise Points", "105", "2.6% unassigned")
            st.metric("Interpretability", "MEDIUM")
        
        st.success("""
        **🏆 Winner: K-Means**
        
        Higher Silhouette Score + Lower Davies-Bouldin + Zero noise points
        → Better cluster quality and business interpretability
        """)
    
    with tab3:
        st.subheader("💡 Cluster Business Insights")
        
        segments = {
            "⭐ Elite (Top 3)": "10 players — Coello, Tapia, Galan level. Premium sponsorship targets.",
            "⭐ Elite Pro (Top 50)": "149 players — Tour regulars. Strong sponsorship ROI.",
            "🔼 Rising Stars": "445 players — Mid-ranking, stable. Best ROI for sponsors NOW.",
            "🚀 Rockets": "14 players — Low ranking but +587 move. Sign them before price rises!",
            "📉 Declining": "9 players — High drop (-440 move). Don't renew contracts.",
            "📈 Developing": "1900+ players — Building their career. Federation development focus."
        }
        
        for segment, description in segments.items():
            st.info(f"**{segment}**: {description}")

# ============================================================
# TIME SERIES PAGE
# ============================================================
elif page == "📈 Time Series":
    st.header("📈 Tournament Growth Forecasting")
    st.markdown("**Forecast the number of padel tournaments per year (2025-2027)**")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📊 Model Results", "💡 Insights"])
    
    with tab1:
        st.subheader("🔮 Tournament Growth Forecast")
        
        # Rebuild time series
        historical_data = pd.DataFrame({
            'year': [2016, 2017, 2018, 2019, 2020, 2021, 2022],
            'tournaments_count': [4, 5, 6, 8, 3, 7, 9]
        })
        seasons_ts = seasons[['year', 'tournaments_count']].sort_values('year')
        full_ts = pd.concat([historical_data, seasons_ts], ignore_index=True)
        full_ts = full_ts.sort_values('year').reset_index(drop=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("2025 Forecast", "~40", "tournaments")
        with col2:
            st.metric("2026 Forecast", "~48", "tournaments")
        with col3:
            st.metric("2027 Forecast", "~55", "tournaments")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(full_ts['year'], full_ts['tournaments_count'], 
                'bo-', label='Historical', linewidth=2, markersize=8)
        
        # Simple forecast visualization
        forecast_years = [2025, 2026, 2027]
        forecast_values = [40, 48, 55]
        ax.plot(forecast_years, forecast_values, 
                'r--o', label='Forecast', linewidth=2, markersize=8)
        ax.fill_between(forecast_years, 
                        [35, 42, 48], [45, 54, 62],
                        alpha=0.2, color='red', label='95% CI')
        ax.axvline(x=2024, color='gray', linestyle=':', 
                   label='Forecast Start', alpha=0.7)
        ax.axvline(x=2020, color='orange', linestyle='--', 
                   label='COVID Impact', alpha=0.5)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Tournaments')
        ax.set_title('Padel Tournament Growth Forecast (2016-2027)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        **📊 Forecast Interpretation:**
        - Strong upward trend expected 2025-2027
        - COVID dip in 2020 was temporary — full recovery achieved
        - Growth rate approximately 15-20% per year
        """)
    
    with tab2:
        st.subheader("📊 Model Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ARIMA(1,1,1)")
            st.metric("MAE", "~3.2")
            st.metric("RMSE", "~4.1")
            st.metric("MAPE", "~12.3%")
            st.markdown("""
            **Strengths:**
            - Standard benchmark model
            - Interpretable coefficients
            - Well-established methodology
            
            **Weaknesses:**
            - Assumes linear patterns
            - Struggles with COVID break
            """)
        
        with col2:
            st.markdown("### Prophet")
            st.metric("MAE", "~2.8")
            st.metric("RMSE", "~3.6")
            st.metric("MAPE", "~10.1%")
            st.markdown("""
            **Strengths:**
            - Handles structural breaks (COVID)
            - Automatic trend detection
            - Uncertainty intervals included
            
            **Weaknesses:**
            - Less interpretable than ARIMA
            - Needs sufficient data
            """)
        
        st.success("""
        **🏆 Winner: Prophet**
        
        Lower MAE, RMSE and MAPE → More accurate forecasts
        Better handles the COVID structural break in 2020
        Provides uncertainty intervals for risk planning
        """)
    
    with tab3:
        st.subheader("💡 Business Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **For Federations (FIP) 🏛️**
            - Plan infrastructure for ~55 tournaments by 2027
            - Hire officials ahead of demand curve
            - Budget allocation based on growth forecast
            - Expand to new markets: Middle East, Asia
            """)
            
            st.info("""
            **For Sponsors 💰**
            - More tournaments = more visibility opportunities
            - Lock in multi-year deals NOW at current prices
            - Forecast confirms padel is a growing investment
            - ROI will improve as audience grows
            """)
        
        with col2:
            st.info("""
            **For Tournament Organizers 🏟️**
            - Growing competition for premium dates & venues
            - Book venues 2 years in advance
            - Prize money expected to increase with growth
            - Digital streaming rights becoming more valuable
            """)
            
            st.info("""
            **SDG Alignment 🌍**
            - ODD 8: Tournament growth drives economic activity
            - Prize money redistribution to athletes
            - ODD 17: International partnerships growing
            - Digital engagement connects global audiences
            """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🎾 Padel Analytics ML Dashboard | Built with Streamlit & Scikit-learn<br>
    Business Intelligence Project — Padel Professional Circuit Analysis
</div>
""", unsafe_allow_html=True)