import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from datetime import datetime
import os

print(f"[{datetime.now()}] Retraining started...")

df = pd.read_csv('data/clean_matches.csv')

le_cat   = LabelEncoder().fit(df['category'])
le_rn    = LabelEncoder().fit(df['round_name'])
le_court = LabelEncoder().fit(df['court'].fillna('Unknown'))
le_src   = LabelEncoder().fit(df['competition_source'])

df['category_enc']   = le_cat.transform(df['category'])
df['round_name_enc'] = le_rn.transform(df['round_name'])
df['court_enc']      = le_court.transform(df['court'].fillna('Unknown'))
df['source_enc']     = le_src.transform(df['competition_source'])
df['month']          = pd.to_datetime(df['played_at'], errors='coerce').dt.month.fillna(1).astype(int)

features = ['category_enc', 'round', 'round_name_enc', 'index', 'court_enc', 'source_enc', 'month']
X = df[features]
y = df['team1_won']

model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

joblib.dump(model, 'models/xgb_classifier_matches.pkl')

os.makedirs('outputs', exist_ok=True)
with open('outputs/retrain_log.txt', 'a') as f:
    f.write(f"[{datetime.now()}] Retrained on {len(df)} samples\n")

print(f"[{datetime.now()}] Done!")
