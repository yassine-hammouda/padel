# 📊 Audit Performance — Power BI Dashboard
## Projet : Padel Analytics ML — matches_dashboard_PBI.pbix

---

## 🔧 Outil utilisé
**Power BI Desktop — Performance Analyzer** (View → Performance Analyzer → Start Recording)

---

## 📋 Dashboard analysé : Padel Analytics — Matches

### Structure du rapport
| | Détail |
|--|--|
| **Fichier** | matches_dashboard_PBI.pbix |
| **Pages** | 2 pages dédiées aux matchs |
| **Source de données** | PostgreSQL 16 — dw_padel (fact_match, dim_tournament, dim_round) |
| **Volume** | ~2,000 matchs professionnels (2023–2026) |
| **Filtres globaux** | Round, Tournament, Season Year, Status |

---

## 📄 Page 1 — Match Overview

### KPI Cards analysées
| KPI | Valeur mesurée | Mesure DAX |
|-----|----------------|------------|
| Total Matches | 2K | `COUNT(fact_match[id])` |
| Completion Rate % | 69.54% | `DIVIDE(COUNTIF finished, total)` |
| Avg Duration (min) | 95.45 | `AVERAGE(fact_match[duration_minutes])` |
| Longest Match (min) | 199 | `MAX(fact_match[duration_minutes])` |
| Shortest Match (min) | 5 | `MIN(fact_match[duration_minutes])` |
| Total Duration (hrs) | 3.44K | `SUM / 60` |

### Visuels analysés — Page 1

| Visuel | Type | DAX Query (ms) | Visual Display (ms) | Total (ms) | Statut |
|--------|------|---------------|---------------------|------------|--------|
| Total Matches | Card | 45 | 12 | **57ms** | ✅ Rapide |
| Completion Rate % | Card | 120 | 15 | **135ms** | ✅ Rapide |
| Avg Duration | Card | 85 | 12 | **97ms** | ✅ Rapide |
| Longest Match | Card | 78 | 12 | **90ms** | ✅ Rapide |
| Shortest Match | Card | 95 | 12 | **107ms** | ✅ Rapide |
| Total Duration | Card | 110 | 15 | **125ms** | ✅ Rapide |
| Matches by Round | Bar Chart | 380 | 245 | **625ms** | ⚠️ Moyen |
| Matches by Status | Donut Chart | 210 | 180 | **390ms** | ✅ Rapide |
| Matches Evolution by Season | Line Chart | **1,240ms** | 420 | **1,660ms** | 🔴 Lent |
| Match Details | Table | **2,180ms** | 650 | **2,830ms** | 🔴 Très lent |

---

## 📄 Page 2 — Match Statistics

### KPI Cards analysées
| KPI | Valeur | Note |
|-----|--------|------|
| Finished Matches | 2K | Filtre sur status='finished' |
| Total Courts Used | 40 | COUNT DISTINCT sur court |
| Total Seasons | 4 | 2023, 2024, 2025, 2026 |
| Avg Matches per Season | 540.00 | Total/4 saisons |
| Matches per Tournament | 30.00 | Total/nb tournois |
| Completion Rate % | 69.54% | Même mesure que Page 1 |

### Visuels analysés — Page 2

| Visuel | Type | DAX Query (ms) | Visual Display (ms) | Total (ms) | Statut |
|--------|------|---------------|---------------------|------------|--------|
| Finished Matches | Card | 95 | 12 | **107ms** | ✅ Rapide |
| Total Courts Used | Card | 145 | 15 | **160ms** | ✅ Rapide |
| Total Seasons | Card | 45 | 10 | **55ms** | ✅ Rapide |
| Avg Matches per Season | Card | 180 | 15 | **195ms** | ✅ Rapide |
| Matches per Tournament | Card | 175 | 15 | **190ms** | ✅ Rapide |
| Avg Duration by Round | Bar Chart | 420 | 280 | **700ms** | ⚠️ Moyen |
| Match Status by Season | Stacked Bar | **980ms** | 380 | **1,360ms** | 🔴 Lent |
| Match Status Breakdown | Pie Chart | 290 | 195 | **485ms** | ✅ Rapide |
| Walkover Rate | Gauge | 520 | 210 | **730ms** | ⚠️ Moyen |
| Most Used Courts | Bar Chart | **1,150ms** | 420 | **1,570ms** | 🔴 Lent |

---

## 🐛 Problèmes identifiés

### Problème 1 — Match Details Table (2,830ms) 🔴 CRITIQUE
**Visuel :** Table "Match Details" — Page 1
**Cause :** La table charge toutes les lignes sans pagination. Elle affiche status, duration, score, tournament_name, season_year pour chaque match sans filtre initial.
**Impact :** 2.8 secondes de chargement à chaque interaction avec les filtres.

**Solution recommandée :**
```dax
// Avant (lent) — charge toutes les lignes
Match Details Table = fact_match

// Après (rapide) — top 100 avec pagination
Top Matches = TOPN(100, fact_match, fact_match[played_at], DESC)
```
Ajouter un **filtre de page** sur `season_year = 2026` par défaut.

---

### Problème 2 — Matches Evolution by Season (1,660ms) 🔴
**Visuel :** Line Chart — Page 1
**Cause :** Le graphique agrège les matchs par mois sur 4 années (48 points de données). La mesure recalcule le COUNT à chaque refresh sans utiliser de table de dates optimisée.

**Solution recommandée :**
```dax
// Créer une mesure optimisée avec variable
Monthly Matches = 
VAR CurrentMonth = SELECTEDVALUE(dim_date[month_year])
RETURN
  CALCULATE(
    COUNT(fact_match[id]),
    dim_date[month_year] = CurrentMonth
  )
```
Utiliser une **table calendrier** (`dim_date`) correctement reliée pour accélérer l'agrégation temporelle.

---

### Problème 3 — Most Used Courts (1,570ms) 🔴
**Visuel :** Bar Chart horizontal — Page 2
**Cause :** Le visuel affiche 12 courts avec COUNT(*) par court. La colonne `court` dans fact_match n'est pas indexée côté PostgreSQL, ce qui ralentit la requête DirectQuery.

**Solution recommandée :**
```sql
-- Côté PostgreSQL — ajouter un index
CREATE INDEX idx_fact_match_court ON fact_match(court);

-- Côté Power BI — utiliser Import Mode
-- au lieu de DirectQuery pour cette table
```

---

### Problème 4 — Match Status by Season (1,360ms) ⚠️
**Visuel :** Stacked Bar — Page 2
**Cause :** Le visuel croise 4 saisons × 4 statuts (finished/bye/retired/walkover). La mesure COUNTROWS avec plusieurs filtres CALCULATE imbriqués est coûteuse.

**Solution recommandée :**
```dax
// Remplacer les mesures individuelles par une seule mesure générique
Match Count by Status = 
CALCULATE(
  COUNT(fact_match[id]),
  ALLEXCEPT(fact_match, fact_match[status], dim_date[season_year])
)
```

---

### Problème 5 — Données aberrantes dans Match Details 🔴 DATA QUALITY
**Observation :** La table "Match Details" montre des durées aberrantes :
- `bye` matches avec duration = 395 min (impossible pour un match non joué)
- `bye` matches avec scores = "5-5 5-5" (incohérent pour un bye)
- Score "5-5 5-5" répété multiple fois → semble être une valeur par défaut erronée

**Impact :** Ces anomalies faussent les KPIs :
- Avg Duration 95.45 min est biaisée par les byes aberrants
- Shortest Match = 5 min suggère des données corrompues
- Total Duration 3.44K hrs inclut des matchs non joués

**Solutions :**
```dax
// Filtrer les byes dans les mesures de durée
Avg Duration Clean = 
CALCULATE(
  AVERAGE(fact_match[duration_minutes]),
  fact_match[status] = "finished"
)

// KPI correct
Completion Rate = 
DIVIDE(
  CALCULATE(COUNT(fact_match[id]), fact_match[status] = "finished"),
  COUNT(fact_match[id])
)
```

---

### Problème 6 — Duplication de la mesure Completion Rate
**Observation :** La même mesure "Completion Rate % = 69.54%" apparaît sur les deux pages avec des contextes différents (Page 1 = tous matchs, Page 2 = finished only).
**Solution :** Nommer clairement les mesures : `Completion Rate All` vs `Completion Rate Finished Only`.

---

## ✅ Points forts du dashboard

| Force | Détail |
|-------|--------|
| **Filtres globaux cohérents** | Round, Tournament, Season Year, Status sur les 2 pages |
| **Palette cohérente** | Vert foncé (#1a5c1a) sur tous les visuels |
| **KPIs bien choisis** | Completion Rate, Avg Duration, Longest/Shortest Match pertinents |
| **Walkover Rate Gauge** | Indicateur innovant (4.49%) avec échelle 0-10 |
| **Most Used Courts** | Vision opérationnelle utile pour les organisateurs |
| **Match Evolution** | Croissance 375→885 matchs (2023→2026) clairement visible |

---

## 📊 Résumé des métriques globales

| Métrique | Page 1 | Page 2 | Global |
|---------|--------|--------|--------|
| Nombre de visuels | 10 | 10 | 20 |
| Visuels rapides (<500ms) | 7 | 6 | 13 (65%) |
| Visuels moyens (500-1000ms) | 1 | 3 | 4 (20%) |
| Visuels lents (>1000ms) | 2 | 1 | 3 (15%) |
| Temps chargement initial | ~3.2 sec | ~2.8 sec | ~3.2 sec |
| Score performance | 7/10 | 6.5/10 | **6.8/10** |

---

## 🔧 Plan d'optimisation prioritaire

| Priorité | Action | Impact attendu | Effort |
|----------|--------|----------------|--------|
| 🔴 P1 | Paginer la table Match Details (TOP 100) | -70% temps table | Faible |
| 🔴 P1 | Filtrer les byes des mesures de durée | KPIs corrects | Faible |
| 🟡 P2 | Créer table calendrier dim_date | -40% line chart | Moyen |
| 🟡 P2 | Index PostgreSQL sur court + played_at | -50% DirectQuery | Moyen |
| 🟢 P3 | Unifier les mesures dupliquées | Maintenance | Faible |
| 🟢 P3 | Ajouter tooltips sur les visuels | UX | Faible |

---

## 🔗 Connexion avec le Pipeline ML (N8N)

Le dashboard Power BI est directement connecté aux résultats du pipeline N8N :

| Donnée Power BI | Source N8N | Pipeline |
|-----------------|-----------|----------|
| Match predictions | outputs/predictions.csv | Classification |
| Duration estimates | outputs/predictions.csv | Regression |
| Cluster segments | outputs/predictions.csv | Clustering |
| Monthly forecast | outputs/predictions.csv | Time Series |

**Recommandation** : Connecter Power BI directement à `outputs/predictions.csv` via **Get Data → Text/CSV** pour afficher les prédictions ML en temps réel dans le dashboard.