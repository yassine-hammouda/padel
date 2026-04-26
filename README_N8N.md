# 🎾 N8N ML Automation Pipeline — Padel Analytics

> **Automated ML inference, monitoring, alerting and retraining for Padel match predictions**

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    N8N ML AUTOMATION PLATFORM                       │
│                                                                     │
│  ⏰ Cron / 🔔 Webhook                                               │
│       ↓                                                             │
│  ❤️  Health Check (API liveness probe)                              │
│       ↓                                                             │
│  📥  Data Retrieval (Flask API)                                     │
│       ↓                                                             │
│  🔄  Feature Engineering (Code Node)                                │
│       ↓                                                             │
│  🤖  ML Model Inference (HTTP POST → Flask)                         │
│       ↓                                                             │
│  📊  Result Enrichment + Confidence Scoring                         │
│       ↓                ↓                                            │
│  💾  Save CSV      ⚠️  Error/Alert Detection                        │
│       ↓                ↓                                            │
│  🔁  Retrain      📋  Log + Email Alert                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 4 Workflows Implemented

### 1. 🏆 Classification Pipeline — Match Winner Prediction
**File:** `n8n_workflow_classification.json`
**Schedule:** Every hour (Cron) + On-demand (Webhook)
**Model:** XGBoost Classifier (74.2% accuracy, AUC 0.79)

| Node | Type | Role |
|------|------|------|
| ⏰ Every Hour Trigger | Schedule Trigger | Auto-lance toutes les heures |
| 🔔 Webhook Trigger | Webhook | Lance manuellement via POST |
| ❤️ Health Check API | HTTP GET | Vérifie que l'API Flask est vivante |
| ✅ API Alive? | IF Node | Branche si API OK ou KO |
| 📥 Get Latest Matches | HTTP GET | Récupère 5 derniers matchs |
| 🔄 Prepare Match Features | Code Node | Encode category, round, court, month... |
| 🤖 XGBoost Predict Winner | HTTP POST | Appelle /predict avec les features |
| 📊 Format + Confidence Score | Code Node | Ajoute score confiance + alertes |
| 💾 Save to CSV via API | Code Node | POST /save_predictions |
| ⚠️ Error Detected? | IF Node | Détecte status == error |
| 📋 Log Error to File | Code Node | POST /log_error |
| 🔁 Auto Retrain XGBoost | HTTP POST | Déclenche /retrain automatiquement |
| 📧 Build Email Report | Code Node | Génère rapport email horlaire |

**Innovations :**
- Health check avant chaque run (liveness probe)
- Score de confiance calculé dynamiquement (< 60% → alerte)
- Dual trigger : Cron + Webhook pour flexibilité
- Auto-retraining après chaque batch de prédictions

---

### 2. ⏱️ Regression Pipeline — Match Duration Prediction
**File:** `n8n_workflow_regression.json`
**Schedule:** Every 6 hours + On-demand (Webhook)
**Model:** Random Forest Regressor (MAE = 26.3 min)

| Node | Type | Role |
|------|------|------|
| ⏰ Every 6 Hours Trigger | Schedule Trigger | Run 4 fois par jour |
| 🔔 Webhook | Webhook | Trigger manuel |
| 📥 Fetch Latest Matches | HTTP GET | Récupère matchs |
| 🔄 Prepare Regression Features | Code Node | Ajoute winner_enc, duration_filled |
| 🤖 RF Predict Duration | HTTP POST | Appelle /predict_duration |
| 📊 Categorize Duration | Code Node | Classe : Short / Standard / Long / Marathon |
| 🚨 Long Match Alert? | IF Node | Alerte si durée > 150 min |
| 📋 Log Long Match Alert | Code Node | Log l'anomalie détectée |
| 💾 Save Duration Predictions | Code Node | Sauvegarde résultats |

**Innovations :**
- Catégorisation intelligente des durées (4 niveaux)
- Alerte automatique sur matchs anormalement longs
- Détection d'anomalies intégrée dans le pipeline

---

### 3. 🗂️ Clustering Pipeline — Match Segment Detection + Drift
**File:** `n8n_workflow_clustering.json`
**Schedule:** Daily + On-demand (Webhook)
**Model:** KMeans k=2 (Silhouette = 0.216)

| Node | Type | Role |
|------|------|------|
| ⏰ Daily Trigger | Schedule Trigger | Run quotidien |
| 🔔 Webhook | Webhook | Trigger manuel |
| 📥 Get Matches Batch | HTTP GET | Récupère batch de matchs |
| 🔄 Prepare Cluster Features | Code Node | Ajoute duration_filled, winner_enc |
| 🤖 KMeans Assign Cluster | HTTP POST | Appelle /cluster |
| 📊 Enrich Cluster Results | Code Node | Nomme les clusters + stats |
| 🔍 Drift Detection | Code Node | Compare distribution vs baseline 64.2%/35.8% |
| ⚠️ Drift Detected? | IF Node | Si écart > 10% → drift alert |
| 📋 Log Drift Alert | Code Node | Log + alerte |
| 🔁 Retrain on Drift | HTTP POST | Réentraîne si drift détecté |
| 💾 Save Cluster Results | Code Node | Sauvegarde segments |

**Innovations :**
- **Drift Detection automatique** — compare distribution quotidienne vs baseline
- **Retraining conditionnel** — déclenché seulement si drift > 10%
- Profiling des clusters enrichi avec noms métier

---

### 4. 📈 Time Series Pipeline — Monthly Volume Forecast + Trend Alert
**File:** `n8n_workflow_timeseries.json`
**Schedule:** Weekly + On-demand (Webhook)
**Model:** ARIMA (MAE = 360 matches/month)

| Node | Type | Role |
|------|------|------|
| ⏰ Weekly Trigger | Schedule Trigger | Rapport hebdomadaire |
| 🔔 Webhook | Webhook | Trigger manuel |
| 📥 Get Recent Matches | HTTP GET | Récupère matchs récents |
| 📊 Aggregate Monthly Volume | Code Node | Agrège par mois + calcule growth rate |
| 🤖 ARIMA Forecast | HTTP POST | Prévision 3 mois |
| 📈 Enrich Forecast Analysis | Code Node | Labels de tendance + recommandations |
| 📉 Declining Trend? | IF Node | Alerte si croissance < -10% |
| 🚨 Alert Volume Decline | Code Node | Log alerte déclin |
| 💾 Save Forecast Results | Code Node | Sauvegarde prévisions |
| 📧 Build Weekly Email Report | Code Node | Rapport hebdomadaire complet |

**Innovations :**
- **Détection automatique de tendance** (Growing/Stable/Declining)
- **Rapport email hebdomadaire** avec prévisions sur 3 mois
- Growth rate calculé dynamiquement entre mois consécutifs

---

## 🔌 API Flask Endpoints

| Endpoint | Méthode | Rôle | Workflow |
|----------|---------|------|----------|
| `/health` | GET | Liveness probe | Tous |
| `/matches/latest` | GET | Récupère 5 derniers matchs | Tous |
| `/predict` | POST | Prédit le vainqueur (XGBoost) | Classification |
| `/predict_duration` | POST | Prédit la durée (RF) | Regression |
| `/cluster` | POST | Assigne un cluster (KMeans) | Clustering |
| `/forecast` | POST | Prévision ARIMA 3 mois | Time Series |
| `/save_predictions` | POST | Sauvegarde CSV | Tous |
| `/retrain` | POST | Réentraîne XGBoost | Classification + Clustering |
| `/log_error` | POST | Log erreurs + alertes | Tous |

---

## 📁 Fichiers générés automatiquement

```
outputs/
├── predictions.csv        ← Prédictions Classification (match_id, winner, proba, timestamp)
├── retrain_log.txt        ← Log de chaque retraining (timestamp + nb samples)
└── pipeline_logs.txt      ← Toutes les erreurs et alertes (JSON lines)
```

---

## 🔔 Système d'alertes

| Alerte | Condition | Action |
|--------|-----------|--------|
| API Down | Health check échoue | Log erreur + skip run |
| Low Confidence | Proba < 60% | Tag "⚠️ LOW CONFIDENCE" |
| Long Match | Durée prédite > 150 min | Log alert |
| Cluster Drift | Distribution écart > 10% | Log + Retrain automatique |
| Volume Decline | Croissance < -10% | Log alert + email |

---

## 🚀 Comment lancer

```powershell
# Terminal 1 — API Flask
cd C:\Users\hammo\Downloads\BI_ML_Project
C:\Users\hammo\AppData\Local\Programs\Python\Python312\python.exe api.py

# Terminal 2 — N8N
$env:NODE_FUNCTION_ALLOW_BUILTIN="fs,path"
D:\npm-global\n8n.cmd start

# Terminal 3 — Dashboard Streamlit
C:\Users\hammo\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app.py
```

| Service | URL |
|---------|-----|
| Dashboard Streamlit | http://localhost:8501 |
| N8N Workflows | http://localhost:5678 |
| Flask API | http://localhost:5000/health |

---

## 📊 Importer les workflows dans N8N

1. Ouvre http://localhost:5678
2. Clique sur **"+"** → **"Import from File"**
3. Sélectionne chaque fichier JSON :
   - `n8n_workflow_classification.json`
   - `n8n_workflow_regression.json`
   - `n8n_workflow_clustering.json`
   - `n8n_workflow_timeseries.json`
4. Clique **"Activate"** (toggle en haut à droite) sur chaque workflow