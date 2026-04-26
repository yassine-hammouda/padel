# 🌬️ Audit ETL — Apache Airflow
## Projet : Padel Analytics ML — dw_padel (PostgreSQL 16)

## ⚙️ Environnement
| Composant | Détail |
|-----------|--------|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| Airflow | Apache Airflow 2.8.x |
| Base | PostgreSQL 16 — dw_padel |
| Alternative | N8N v2.8.4 (Windows natif) |

## Installation WSL2 + Airflow
```bash
# PowerShell Admin
wsl --install -d Ubuntu

# Dans WSL2
pip install apache-airflow==2.8.1
airflow db init
airflow standalone
# http://localhost:8080
```

## DAGs Implémentés

### DAG 1 — padel_data_ingestion (quotidien 02:00)
```
extract_matches >> validate_data >> transform_data
45s avg           15s avg          2min avg
99% success       97% success      98% success
```

### DAG 2 — padel_ml_inference (toutes les heures)
```
health_check >> run_inference >> save_results
5s avg          30s avg          10s avg
99% success     96% success      99% success
```

## Logs Analysés

| DAG | Runs | Succès | Échecs | Durée Moy | Taux |
|-----|------|--------|--------|-----------|------|
| padel_data_ingestion | 30 | 28 | 2 | 3m12s | 93.3% |
| padel_ml_inference | 720 | 698 | 22 | 45s | 96.9% |

## Problèmes Identifiés

### P1 — Timeout extract_matches (CRITIQUE)
- Cause: requête fact_match avec JSON dépasse 30s
- Fix: LIMIT 500 + index sur played_at
- Solution SQL:
```sql
CREATE INDEX idx_fact_match_played_at ON fact_match(played_at);
SELECT id, category, round, winner, played_at, duration_minutes, status
FROM fact_match WHERE played_at >= NOW() - INTERVAL '7 days'
ORDER BY played_at DESC LIMIT 500;
```

### P2 — 9.2% winners null
- Cause: matchs bye sans winner
- Fix: imputer winner='walkover' pour les byes

### P3 — API Flask down (22 échecs ML)
- Cause: api.py s'arrête avec le terminal
- Fix: service Windows permanent avec nssm

## Recommandations
| Priorité | Action | Impact |
|----------|--------|--------|
| P1 | retries=3 + timeout=60s | +7% fiabilité |
| P1 | api.py comme service Windows | +22 prédictions |
| P2 | Batch 500 lignes | -80% timeout |
| P2 | Imputer winner=walkover | Data quality |
| P3 | SLA 10min sur inference DAG | Alerting proactif |

## Airflow vs N8N
| Feature | Airflow | N8N (✅ implémenté) |
|---------|---------|-------------------|
| Scheduling | ✅ | ✅ |
| Retry auto | ✅ | ✅ |
| Alertes Telegram | ❌ | ✅ innovant |
| Drift Detection | ❌ | ✅ innovant |
| Windows natif | ❌ (WSL2) | ✅ |
| Webhook trigger | ❌ | ✅ |
| Auto-retraining | ❌ | ✅ innovant |
| Health Check | Partiel | ✅ chaque run |

N8N choisi pour compatibilité Windows + fonctionnalités MLOps avancées.