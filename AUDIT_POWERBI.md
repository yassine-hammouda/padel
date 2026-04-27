# 📊 Audit Performance — Power BI Dashboard
## Projet : Padel Analytics ML — matches_dashboard_PBI.pbix
## Date d'audit : 26/04/2026

---

## 🔧 Outil utilisé
**Power BI Desktop — Analyseur de performances**
(Afficher → Analyseur de performances → Démarrer l'enregistrement)

---

## 📋 Dashboard analysé

| | Détail |
|--|--|
| **Fichier** | matches_dashboard_PBI.pbix |
| **Pages** | 2 pages (Page 1 + Doublon de Page 1) |
| **Source** | PostgreSQL 16 — dw_padel |
| **Tables** | public.fact_match, dim_player, dim_tournament, dim_round, dim_date, dim_location, dim_brand, dim_sponsor, dim_team, dim_tour |
| **Volume** | ~2,000 matchs professionnels (2023–2026) |
| **Filtres globaux** | Round, Tournament, Season Year, Status |

---

## 📄 Page 1 — Match Overview

### Mesures réelles (Performance Analyzer)

| Visuel | Durée (ms) | Statut |
|--------|-----------|--------|
| Segment (Round) | **30ms** | ✅ Rapide |
| Segment (Tournament) | **25ms** | ✅ Rapide |
| Segment (Season Year) | **31ms** | ✅ Rapide |
| Segment (Status) | **26ms** | ✅ Rapide |
| Forme (Header) | **120ms** | ✅ Rapide |
| Zone de texte | **72ms** | ✅ Rapide |
| Carte (Total Matches) | **279ms** | ✅ Rapide |
| Carte (Completion Rate) | **281ms** | ✅ Rapide |
| Carte (Avg Duration) | **283ms** | ✅ Rapide |
| Carte (Longest Match) | **286ms** | ✅ Rapide |
| Carte (Shortest Match) | **264ms** | ✅ Rapide |
| Carte (Total Duration) | **384ms** | ✅ Rapide |
| Matches by Round | **379ms** | ✅ Rapide |
| Matches by Status | **376ms** | ✅ Rapide |
| Matches Evolution by Season Year | **376ms** | ✅ Rapide |
| Match Details (Table) | **351ms** | ✅ Rapide |

**🏆 Page 1 — Excellente performance ! Tous les visuels < 500ms**

---

## 📄 Page 2 — Match Statistics (Doublon de Page 1)

### Mesures réelles (Performance Analyzer)

| Visuel | Durée (ms) | Statut |
|--------|-----------|--------|
| Forme (Header) | **163ms** | ✅ Rapide |
| Zone de texte | **142ms** | ✅ Rapide |
| Segment (Round) | **161ms** | ✅ Rapide |
| Segment (Tournament) | **161ms** | ✅ Rapide |
| Segment (Season Year) | **160ms** | ✅ Rapide |
| Segment (Status) | **160ms** | ✅ Rapide |
| Match Status Breakdown | **245ms** | ✅ Rapide |
| Avg Duration by Round | **245ms** | ✅ Rapide |
| Match Status by Season | **243ms** | ✅ Rapide |
| Walkover Rate | **241ms** | ✅ Rapide |
| Most Used Courts | **241ms** | ✅ Rapide |
| Carte (Finished Matches) | **282ms** | ✅ Rapide |
| Carte (Total Courts) | **284ms** | ✅ Rapide |
| Carte (Total Seasons) | **287ms** | ✅ Rapide |
| Carte (Avg Matches/Season) | **289ms** | ✅ Rapide |
| Carte (Matches/Tournament) | **292ms** | ✅ Rapide |
| Carte (Completion Rate) | **216ms** | ✅ Rapide |

**🏆 Page 2 — Excellente performance ! Tous les visuels < 500ms**

---

## 📊 Analyse des interactions (avec filtres actifs)

Lors des interactions avec les filtres, les temps augmentent légèrement :

| Visuel | Sans filtre | Avec filtre | Variation |
|--------|------------|-------------|-----------|
| Matches by Round | 379ms | **447ms** | +18% ⚠️ |
| Matches by Status | 376ms | **405ms** | +8% ✅ |
| Matches Evolution | 376ms | **445ms** | +18% ⚠️ |
| Match Details | 351ms | **441ms** | +26% ⚠️ |
| Most Used Courts | 241ms | **408ms** | +69% ⚠️ |

---

## 🐛 Problèmes identifiés

### Problème 1 — Match Details Table (351–441ms selon filtre) ⚠️
**Cause :** La table charge toutes les lignes sans pagination.
Elle affiche status, duration, score, tournament_name, season_year.
**Impact :** Augmente de 26% lors des interactions avec les filtres.

**Solution DAX :**
```dax
Top Matches = TOPN(100, fact_match, fact_match[played_at], DESC)
```
Ajouter un filtre de page sur `season_year = 2026` par défaut.

---

### Problème 2 — Most Used Courts (241→408ms) ⚠️
**Cause :** +69% de temps lors des interactions filtres.
La colonne `court` dans fact_match n'est pas indexée côté PostgreSQL.

**Solution SQL :**
```sql
CREATE INDEX idx_fact_match_court ON fact_match(court);
CREATE INDEX idx_fact_match_played_at ON fact_match(played_at);
```

---

### Problème 3 — Données aberrantes dans Match Details 🔴
**Observation :** Matchs `bye` avec :
- Duration = 395 min (impossible pour un match non joué)
- Score = "5-5 5-5" (incohérent pour un bye)
- Total duration affiché : **22,610 min** incluant les byes

**Impact sur KPIs :**
- Avg Duration 99.60 min biaisée par les byes
- Shortest Match = 45 min (valeur suspecte)
- Total Duration 376.83 hrs inclut des matchs non joués

**Solution DAX :**
```dax
Avg Duration Clean =
CALCULATE(
  AVERAGE(fact_match[duration_minutes]),
  fact_match[status] = "finished"
)

Match Count Clean =
CALCULATE(
  COUNT(fact_match[id]),
  fact_match[status] IN {"finished", "retired"}
)
```

---

### Problème 4 — Duplication mesure Completion Rate ⚠️
**Observation :** Completion Rate 69.54% apparaît sur les 2 pages avec contextes différents.
**Solution :** Créer 2 mesures distinctes :
- `Completion Rate All` (Page 1)
- `Completion Rate Finished` (Page 2)

---

### Problème 5 — Filtres sans valeur par défaut ⚠️
**Observation :** Les 4 filtres (Round, Tournament, Season, Status) affichent "Tout" par défaut → charge toutes les données au démarrage.
**Solution :** Définir `Season Year = 2025` comme filtre par défaut pour réduire le volume initial.

---

## ✅ Points forts du dashboard

| Force | Détail |
|-------|--------|
| **Performance excellente** | 100% des visuels < 500ms sans filtre |
| **Filtres globaux cohérents** | Round, Tournament, Season Year, Status sur les 2 pages |
| **Palette cohérente** | Vert foncé (#1a5c1a) sur tous les visuels |
| **KPIs bien choisis** | Completion Rate, Avg Duration, Longest/Shortest pertinents |
| **Walkover Rate Gauge** | Indicateur innovant (4.49%) avec échelle 0-10 |
| **Most Used Courts** | Vision opérationnelle utile pour les organisateurs |
| **Match Evolution** | Croissance 375→885 matchs (2023→2026) visible |
| **Modèle de données riche** | 12 tables connectées (dim + fact) |

---

## 📊 Résumé des métriques globales

| Métrique | Page 1 | Page 2 | Global |
|---------|--------|--------|--------|
| Nombre de visuels | 16 | 17 | 33 |
| Visuels rapides (<500ms) | 16 (100%) | 17 (100%) | 33 (100%) |
| Visuels moyens (500-1000ms) | 0 | 0 | 0 |
| Visuels lents (>1000ms) | 0 | 0 | 0 |
| Temps chargement initial | ~380ms | ~290ms | ~380ms |
| Score performance | **9.5/10** | **9.5/10** | **9.5/10** |

---

## 🔧 Plan d'optimisation prioritaire

| Priorité | Action | Impact attendu | Effort |
|----------|--------|----------------|--------|
| 🔴 P1 | Filtrer les byes des mesures durée | KPIs corrects | Faible |
| 🔴 P1 | Index PostgreSQL sur court + played_at | -30% filtres | Moyen |
| 🟡 P2 | Paginer Match Details (TOP 100) | -26% interactions | Faible |
| 🟡 P2 | Filtre par défaut Season Year = 2025 | Chargement initial | Faible |
| 🟢 P3 | Séparer mesures Completion Rate | Lisibilité | Faible |
| 🟢 P3 | Ajouter tooltips sur visuels | UX | Faible |

---

## 🔗 Connexion Pipeline ML → Power BI

| Donnée Power BI | Source N8N | Pipeline ML |
|-----------------|-----------|-------------|
| Match predictions | outputs/predictions.csv | Classification XGBoost (74.2%) |
| Duration estimates | outputs/predictions.csv | RF Regressor (MAE=26.3 min) |
| Cluster segments | outputs/predictions.csv | KMeans k=2 (Silhouette=0.216) |
| Monthly forecast | outputs/predictions.csv | ARIMA (MAE=360/mois) |

**Recommandation :** Connecter Power BI à `outputs/predictions.csv` via
**Obtenir les données → Texte/CSV** pour visualiser les prédictions ML en temps réel.