from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.http import SimpleHttpOperator
import pandas as pd
import requests

default_args = {
    'owner': 'hammo',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': True,
    'email': ['hammo@example.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'dag_padel_prediction',
    default_args=default_args,
    description='Pipeline ML automatique pour prediction matchs Padel',
    schedule_interval='@hourly',
    catchup=False,
    tags=['padel', 'ml', 'prediction'],
)

def extract_matches(**context):
    response = requests.get('http://localhost:5000/matches/latest')
    matches = response.json()
    context['ti'].xcom_push(key='matches', value=matches)
    print(f"Extracted {len(matches)} matches")

def predict_matches(**context):
    matches = context['ti'].xcom_pull(key='matches', task_ids='extract_matches')
    predictions = []
    for match in matches:
        payload = {
            'category': match.get('category', 'men'),
            'round': match.get('round', 1),
            'round_name': match.get('round_name', 'Finals'),
            'index': match.get('index', 0),
            'court': match.get('court', 'Unknown'),
            'competition_source': match.get('competition_source', 'FIP'),
            'month': datetime.now().month
        }
        response = requests.post('http://localhost:5000/predict', json=payload)
        pred = response.json()
        pred['match_id'] = match.get('id', 'N/A')
        predictions.append(pred)
    context['ti'].xcom_push(key='predictions', value=predictions)
    print(f"Predicted {len(predictions)} matches")

def save_predictions(**context):
    predictions = context['ti'].xcom_pull(key='predictions', task_ids='predict_matches')
    formatted = [{
        'match_id': p.get('match_id'),
        'predicted_winner': p.get('winner'),
        'probability_team1': p.get('probability_team1'),
        'probability_team2': p.get('probability_team2'),
        'timestamp': datetime.now().isoformat()
    } for p in predictions]
    requests.post('http://localhost:5000/save_predictions', json={'predictions': formatted})
    print(f"Saved {len(formatted)} predictions")

def retrain_model(**context):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv('/opt/airflow/data/clean_matches.csv')
    features = ['category_enc', 'round', 'round_name_enc', 'index', 'court_enc', 'source_enc', 'month']
    X = df[features]
    y = df['team1_won']
    model = joblib.load('/opt/airflow/models/xgb_classifier_matches.pkl')
    model.fit(X, y)
    joblib.dump(model, '/opt/airflow/models/xgb_classifier_matches.pkl')
    print("Model retrained successfully")

t1 = PythonOperator(
    task_id='extract_matches',
    python_callable=extract_matches,
    dag=dag,
)

t2 = PythonOperator(
    task_id='predict_matches',
    python_callable=predict_matches,
    dag=dag,
)

t3 = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions,
    dag=dag,
)

t4 = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
    trigger_rule='all_done',
)

t1 >> t2 >> t3 >> t4
