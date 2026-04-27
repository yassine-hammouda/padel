from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import csv
import os
import json
import tempfile

warnings.filterwarnings('ignore')

# ── Force temp directory to D: to avoid C: space issues ──────────────
os.environ['TEMP'] = 'D:\\tmp'
os.environ['TMP']  = 'D:\\tmp'
os.makedirs('D:\\tmp', exist_ok=True)
tempfile.tempdir = 'D:\\tmp'

# ── Base directory (projet) ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

model = joblib.load('D:/padel_models/xgb_classifier_matches.pkl')
df_ref   = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_matches.csv'))

le_cat   = LabelEncoder().fit(df_ref['category'])
le_rn    = LabelEncoder().fit(df_ref['round_name'])
le_court = LabelEncoder().fit(df_ref['court'].fillna('Unknown'))
le_src   = LabelEncoder().fit(df_ref['competition_source'])


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'XGBoost Matches Classifier'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        row = pd.DataFrame([{
            'category_enc':   le_cat.transform([data['category']])[0],
            'round':          data['round'],
            'round_name_enc': le_rn.transform([data['round_name']])[0],
            'index':          data['index'],
            'court_enc':      le_court.transform([data.get('court', 'Unknown')])[0],
            'source_enc':     le_src.transform([data['competition_source']])[0],
            'month':          data['month']
        }])
        prediction = model.predict(row)[0]
        proba = model.predict_proba(row)[0].tolist()
        return jsonify({
            'prediction':        int(prediction),
            'winner':            'Team 1' if prediction == 1 else 'Team 2',
            'probability_team1': round(proba[1] * 100, 1),
            'probability_team2': round(proba[0] * 100, 1),
            'match_id':          data.get('match_id', 'N/A'),
            'match_name':        data.get('match_name', 'Unknown'),
            'status':            'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


@app.route('/matches/latest', methods=['GET'])
def get_latest_matches():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_matches.csv'))
    latest = df.tail(5).fillna('Unknown')
    records = latest.to_dict(orient='records')
    for r in records:
        r['match_id'] = r.pop('id', 'N/A')
    return jsonify(records)


@app.route('/save_predictions', methods=['POST'])
def save_predictions():
    data = request.json
    predictions = data.get('predictions', [])
    filepath = os.path.join(BASE_DIR, 'outputs', 'predictions.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'match_id', 'predicted_winner',
            'probability_team1', 'probability_team2', 'timestamp'
        ])
        if not file_exists:
            writer.writeheader()
        for p in predictions:
            writer.writerow(p)
    return jsonify({'saved': True, 'rows': len(predictions)})


@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        from xgboost import XGBClassifier
        from datetime import datetime

        # Force temp sur D: pour XGBoost
        os.environ['TEMP'] = 'D:\\tmp'
        os.environ['TMP']  = 'D:\\tmp'
        os.makedirs('D:\\tmp', exist_ok=True)

        df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clean_matches.csv'))
        le_c  = LabelEncoder().fit(df['category'])
        le_r  = LabelEncoder().fit(df['round_name'])
        le_co = LabelEncoder().fit(df['court'].fillna('Unknown'))
        le_s  = LabelEncoder().fit(df['competition_source'])
        df['category_enc']   = le_c.transform(df['category'])
        df['round_name_enc'] = le_r.transform(df['round_name'])
        df['court_enc']      = le_co.transform(df['court'].fillna('Unknown'))
        df['source_enc']     = le_s.transform(df['competition_source'])
        df['month'] = pd.to_datetime(df['played_at'], errors='coerce').dt.month.fillna(1).astype(int)

        features = ['category_enc','round','round_name_enc','index','court_enc','source_enc','month']
        X = df[features]
        y = (df['winner'] == 'team_1').astype(int)

        m = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        m.fit(X, y)

        model_path = 'D:/padel_models/xgb_classifier_matches.pkl'        
        os.makedirs('D:/padel_models', exist_ok=True)

        joblib.dump(m, model_path)

        log_path = 'D:/padel_models/retrain_log.txt'            
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(f'[{datetime.now()}] Retrained on {len(df)} samples\n')

        return jsonify({'retrained': True, 'samples': len(df), 'status': 'success'})
    except Exception as e:
        return jsonify({'retrained': False, 'error': str(e), 'status': 'error'}), 400


@app.route('/log_error', methods=['POST'])
def log_error():
    data = request.json
    log_path = os.path.join(BASE_DIR, 'outputs', 'pipeline_logs.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(data) + '\n')
    return jsonify({'logged': True})


@app.route('/predict_duration', methods=['POST'])
def predict_duration():
    try:
        model_reg = joblib.load('D:/padel_models/rf_regressor_matches.pkl')
        data = request.json
        row = pd.DataFrame([{
            'category_enc':   le_cat.transform([data['category']])[0],
            'round':          data['round'],
            'round_name_enc': le_rn.transform([data['round_name']])[0],
            'index':          data['index'],
            'court_enc':      le_court.transform([data.get('court', 'Unknown')])[0],
            'source_enc':     le_src.transform([data['competition_source']])[0],
            'month':          data['month'],
            'winner_enc':     data.get('winner_enc', 0)
        }])
        duration = model_reg.predict(row)[0]
        return jsonify({
            'predicted_duration_minutes': round(float(duration), 1),
            'match_id':   data.get('match_id', 'N/A'),
            'match_name': data.get('match_name', 'Unknown'),
            'status':     'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        kmeans = joblib.load('D:/padel_models/kmeans_matches.pkl')
        scaler = joblib.load('D:/padel_models/scaler_matches.pkl')
        data = request.json
        row = pd.DataFrame([{
            'category_enc':    le_cat.transform([data['category']])[0],
            'round':           data['round'],
            'round_name_enc':  le_rn.transform([data['round_name']])[0],
            'index':           data['index'],
            'court_enc':       le_court.transform([data.get('court', 'Unknown')])[0],
            'source_enc':      le_src.transform([data['competition_source']])[0],
            'month':           data['month'],
            'winner_enc':      data.get('winner_enc', 0),
            'duration_filled': data.get('duration_filled', 87.0)
        }])
        scaled     = scaler.transform(row)
        cluster_id = kmeans.predict(scaled)[0]
        return jsonify({
            'cluster':      int(cluster_id),
            'cluster_name': f'Cluster {cluster_id}',
            'match_id':     data.get('match_id', 'N/A'),
            'status':       'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        from datetime import datetime, timedelta
        data    = request.json
        series  = data.get('series', [])
        periods = int(data.get('periods', 3))
        if isinstance(series, str):
            series = json.loads(series)
        counts = [int(s['count']) for s in series] if len(series) >= 2 else [100, 120, 110]
        avg    = sum(counts[-3:]) / min(3, len(counts))
        trend  = (counts[-1] - counts[0]) / max(len(counts), 1)
        forecasts = []
        last_date = datetime.now()
        for i in range(1, periods + 1):
            future_date = last_date + timedelta(days=i * 30)
            predicted   = max(0, round(avg + trend * i, 0))
            forecasts.append({'month': str(future_date)[:10], 'predicted': int(predicted)})
        return jsonify({'forecast': forecasts, 'method': 'Moving Average', 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


if __name__ == '__main__':
    print("API Flask demarree sur http://localhost:5000")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"TEMP: {os.environ.get('TEMP')}")
    app.run(host='0.0.0.0', port=5000, debug=False)