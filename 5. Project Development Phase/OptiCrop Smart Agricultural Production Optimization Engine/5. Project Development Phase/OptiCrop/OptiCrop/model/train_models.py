"""
OptiCrop - ML Model Training Pipeline
Trains KNN, Logistic Regression, Decision Tree, Random Forest, K-Means
Selects best model and saves as .pkl
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

np.random.seed(42)

CROPS = [
    'Rice', 'Wheat', 'Maize', 'Chickpea', 'KidneyBeans', 'PigeonPeas',
    'MothBeans', 'MungBeans', 'Blackgram', 'Lentil', 'Pomegranate',
    'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple',
    'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee'
]

CROP_PROFILES = {
    'Rice':         dict(N=(60,100),  P=(35,55),  K=(35,55),  temp=(20,30), hum=(70,90), ph=(5.5,7.0), rain=(150,300)),
    'Wheat':        dict(N=(60,100),  P=(40,60),  K=(40,60),  temp=(10,25), hum=(50,70), ph=(5.5,7.5), rain=(50,100)),
    'Maize':        dict(N=(60,110),  P=(35,60),  K=(40,65),  temp=(18,30), hum=(55,75), ph=(5.5,7.5), rain=(50,100)),
    'Chickpea':     dict(N=(30,60),   P=(55,80),  K=(55,80),  temp=(15,25), hum=(15,40), ph=(6.0,8.0), rain=(30,80)),
    'KidneyBeans':  dict(N=(15,35),   P=(55,80),  K=(15,25),  temp=(15,25), hum=(15,50), ph=(5.5,7.0), rain=(20,60)),
    'PigeonPeas':   dict(N=(15,35),   P=(55,80),  K=(15,25),  temp=(18,30), hum=(30,60), ph=(5.5,7.0), rain=(30,70)),
    'MothBeans':    dict(N=(15,35),   P=(35,55),  K=(30,50),  temp=(25,35), hum=(25,50), ph=(6.0,8.0), rain=(30,70)),
    'MungBeans':    dict(N=(15,35),   P=(35,55),  K=(15,25),  temp=(25,35), hum=(30,60), ph=(6.0,7.5), rain=(30,70)),
    'Blackgram':    dict(N=(30,50),   P=(55,80),  K=(15,25),  temp=(25,35), hum=(30,70), ph=(5.5,7.5), rain=(30,70)),
    'Lentil':       dict(N=(15,35),   P=(55,80),  K=(15,25),  temp=(10,25), hum=(40,65), ph=(5.5,7.5), rain=(30,80)),
    'Pomegranate':  dict(N=(15,35),   P=(15,25),  K=(15,25),  temp=(18,32), hum=(80,95), ph=(5.5,7.5), rain=(100,200)),
    'Banana':       dict(N=(80,120),  P=(55,80),  K=(45,65),  temp=(22,32), hum=(70,90), ph=(5.5,7.0), rain=(100,200)),
    'Mango':        dict(N=(15,25),   P=(15,25),  K=(15,25),  temp=(22,35), hum=(45,65), ph=(5.5,7.5), rain=(50,100)),
    'Grapes':       dict(N=(15,25),   P=(15,25),  K=(35,55),  temp=(8,32),  hum=(60,80), ph=(5.5,7.5), rain=(50,100)),
    'Watermelon':   dict(N=(80,120),  P=(55,80),  K=(40,60),  temp=(24,35), hum=(60,85), ph=(5.5,7.5), rain=(40,80)),
    'Muskmelon':    dict(N=(80,120),  P=(55,80),  K=(40,60),  temp=(24,35), hum=(75,95), ph=(6.0,7.5), rain=(20,40)),
    'Apple':        dict(N=(15,25),   P=(120,145),K=(195,210),temp=(1,20),  hum=(85,95), ph=(5.5,7.0), rain=(100,125)),
    'Orange':       dict(N=(15,25),   P=(15,25),  K=(15,25),  temp=(10,30), hum=(90,95), ph=(6.0,7.5), rain=(100,150)),
    'Papaya':       dict(N=(45,65),   P=(35,55),  K=(45,65),  temp=(25,35), hum=(85,95), ph=(6.0,7.5), rain=(150,200)),
    'Coconut':      dict(N=(15,25),   P=(15,25),  K=(15,25),  temp=(22,35), hum=(85,95), ph=(5.5,7.0), rain=(100,200)),
    'Cotton':       dict(N=(100,140), P=(35,55),  K=(15,25),  temp=(22,36), hum=(55,75), ph=(6.0,8.0), rain=(60,110)),
    'Jute':         dict(N=(60,100),  P=(35,55),  K=(35,55),  temp=(22,36), hum=(60,90), ph=(6.0,7.0), rain=(150,250)),
    'Coffee':       dict(N=(80,120),  P=(15,25),  K=(25,45),  temp=(15,28), hum=(85,95), ph=(3.5,6.5), rain=(100,300)),
}

def generate_dataset(n_per_crop=100):
    rows = []
    for crop, p in CROP_PROFILES.items():
        for _ in range(n_per_crop):
            row = {
                'N':           np.random.uniform(*p['N']),
                'P':           np.random.uniform(*p['P']),
                'K':           np.random.uniform(*p['K']),
                'temperature': np.random.uniform(*p['temp']),
                'humidity':    np.random.uniform(*p['hum']),
                'ph':          np.random.uniform(*p['ph']),
                'rainfall':    np.random.uniform(*p['rain']),
                'label':       crop
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def preprocess(df):
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['label'])
    features = ['N','P','K','temperature','humidity','ph','rainfall']
    X = df[features].values
    y = df['crop_encoded'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, le

def train_all_models(X_train, X_test, y_train, y_test):
    models = {
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42, max_depth=15),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        results[name] = {
            'model':    model,
            'accuracy': round(acc  * 100, 2),
            'f1':       round(f1   * 100, 2),
            'cv_score': round(cv   * 100, 2),
        }
        print(f"  {name:22s} | Acc: {acc*100:.2f}%  F1: {f1*100:.2f}%  CV: {cv*100:.2f}%")
    return results

def run_kmeans(X_scaled, n_clusters=23):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km

if __name__ == '__main__':
    print("═"*60)
    print("  OptiCrop — ML Training Pipeline")
    print("═"*60)

    print("\n[1/5] Generating dataset …")
    df = generate_dataset(n_per_crop=120)
    df.to_csv(os.path.join(DATA_DIR, 'crop_data.csv'), index=False)
    print(f"      {len(df)} records | {df['label'].nunique()} crops")

    print("\n[2/5] Preprocessing …")
    X_scaled, y, scaler, le = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("\n[3/5] Training models …")
    results = train_all_models(X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    print(f"\n[4/5] Best model → {best_name} ({results[best_name]['accuracy']}%)")

    print("\n[5/5] Saving artefacts …")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'crop_model.pkl'),  'wb') as f: pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'),      'wb') as f: pickle.dump(scaler,     f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'),'wb') as f: pickle.dump(le,         f)

    km = run_kmeans(X_scaled)
    with open(os.path.join(MODEL_DIR, 'kmeans.pkl'), 'wb') as f: pickle.dump(km, f)

    model_metrics = {
        k: {'accuracy': v['accuracy'], 'f1': v['f1'], 'cv_score': v['cv_score']}
        for k, v in results.items()
    }
    model_metrics['best_model'] = best_name

    feature_importances = {}
    if hasattr(best_model, 'feature_importances_'):
        features = ['N','P','K','temperature','humidity','ph','rainfall']
        for f, imp in zip(features, best_model.feature_importances_):
            feature_importances[f] = round(float(imp)*100, 2)

    meta = {
        'best_model':          best_name,
        'accuracy':            results[best_name]['accuracy'],
        'f1':                  results[best_name]['f1'],
        'cv_score':            results[best_name]['cv_score'],
        'crops':               CROPS,
        'features':            ['N','P','K','temperature','humidity','ph','rainfall'],
        'model_metrics':       model_metrics,
        'feature_importances': feature_importances,
        'crop_profiles':       {k: {kk: list(vv) for kk, vv in v.items()}
                                for k, v in CROP_PROFILES.items()},
    }
    with open(os.path.join(MODEL_DIR, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n  crop_model.pkl   ✓")
    print("  scaler.pkl       ✓")
    print("  label_encoder.pkl✓")
    print("  kmeans.pkl       ✓")
    print("  model_meta.json  ✓")
    print("\n" + "═"*60)
    print("  Training complete!")
    print("═"*60)
