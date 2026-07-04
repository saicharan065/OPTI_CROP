import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'models')
STATIC_IMG_DIR = os.path.join(BASE, 'static', 'images')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. DATASET GENERATION
# -----------------------------------------------------------
crop_profiles = {
    'rice':        dict(N=(60,100),  P=(30,60),   K=(30,60),   temp=(20,35),  hum=(70,90),  ph=(5.5,7.0), rain=(150,250)),
    'wheat':       dict(N=(80,120),  P=(35,65),   K=(35,65),   temp=(15,25),  hum=(55,75),  ph=(6.0,7.5), rain=(60,120)),
    'maize':       dict(N=(70,110),  P=(35,65),   K=(40,70),   temp=(18,30),  hum=(55,75),  ph=(5.8,7.2), rain=(60,110)),
    'cotton':      dict(N=(90,130),  P=(35,65),   K=(40,70),   temp=(25,35),  hum=(50,70),  ph=(6.0,7.5), rain=(60,110)),
    'sugarcane':   dict(N=(100,140), P=(30,55),   K=(30,55),   temp=(22,35),  hum=(65,85),  ph=(5.5,7.0), rain=(110,200)),
    'banana':      dict(N=(90,130),  P=(65,95),   K=(40,70),   temp=(22,32),  hum=(70,90),  ph=(5.5,6.5), rain=(110,200)),
    'mango':       dict(N=(15,35),   P=(10,30),   K=(25,45),   temp=(24,35),  hum=(45,65),  ph=(5.5,7.5), rain=(50,100)),
    'grapes':      dict(N=(15,35),   P=(10,30),   K=(25,45),   temp=(15,30),  hum=(55,75),  ph=(5.5,7.5), rain=(60,110)),
    'watermelon':  dict(N=(90,120),  P=(10,30),   K=(40,60),   temp=(22,35),  hum=(55,80),  ph=(5.5,7.0), rain=(40,80)),
    'muskmelon':   dict(N=(90,120),  P=(10,30),   K=(40,60),   temp=(25,38),  hum=(85,100), ph=(5.0,6.5), rain=(20,60)),
    'apple':       dict(N=(0,20),    P=(120,145), K=(195,210), temp=(18,24),  hum=(89,97),  ph=(5.5,6.5), rain=(100,125)),
    'orange':      dict(N=(0,20),    P=(5,20),    K=(5,20),    temp=(10,22),  hum=(88,97),  ph=(6.0,7.5), rain=(100,125)),
    'papaya':      dict(N=(45,60),   P=(40,60),   K=(40,60),   temp=(25,35),  hum=(85,100), ph=(6.0,7.0), rain=(120,200)),
    'coconut':     dict(N=(15,30),   P=(5,20),    K=(25,45),   temp=(22,32),  hum=(85,100), ph=(5.0,6.5), rain=(150,250)),
    'jute':        dict(N=(60,90),   P=(35,55),   K=(35,55),   temp=(22,35),  hum=(70,90),  ph=(6.0,7.0), rain=(150,250)),
    'coffee':      dict(N=(90,110),  P=(30,50),   K=(30,50),   temp=(15,25),  hum=(55,75),  ph=(6.0,7.5), rain=(150,250)),
    'chickpea':    dict(N=(30,60),   P=(60,100),  K=(70,120),  temp=(18,28),  hum=(14,22),  ph=(5.5,7.5), rain=(80,130)),
    'kidneybeans': dict(N=(15,35),   P=(120,140), K=(15,30),   temp=(18,26),  hum=(18,24),  ph=(5.5,7.0), rain=(100,150)),
    'pigeonpeas':  dict(N=(15,30),   P=(60,80),   K=(15,30),   temp=(18,28),  hum=(38,55),  ph=(5.0,7.0), rain=(100,200)),
    'mothbeans':   dict(N=(15,30),   P=(40,65),   K=(15,30),   temp=(24,35),  hum=(40,55),  ph=(3.5,6.5), rain=(30,60)),
    'mungbean':    dict(N=(15,30),   P=(40,65),   K=(15,30),   temp=(25,35),  hum=(80,90),  ph=(6.2,7.2), rain=(30,60)),
    'blackgram':   dict(N=(30,50),   P=(60,90),   K=(15,35),   temp=(25,35),  hum=(60,70),  ph=(5.5,7.5), rain=(60,90)),
    'lentil':      dict(N=(15,30),   P=(60,90),   K=(15,30),   temp=(15,25),  hum=(60,70),  ph=(5.5,7.5), rain=(30,70)),
    'pomegranate': dict(N=(15,30),   P=(5,20),    K=(35,55),   temp=(18,35),  hum=(85,100), ph=(5.5,7.0), rain=(100,125)),
}

records = []
n_per_crop = 100
for crop, p in crop_profiles.items():
    N    = np.random.uniform(*p['N'],    n_per_crop)
    P    = np.random.uniform(*p['P'],    n_per_crop)
    K    = np.random.uniform(*p['K'],    n_per_crop)
    temp = np.random.uniform(*p['temp'], n_per_crop)
    hum  = np.random.uniform(*p['hum'],  n_per_crop)
    ph   = np.random.uniform(*p['ph'],   n_per_crop)
    rain = np.random.uniform(*p['rain'], n_per_crop)
    for i in range(n_per_crop):
        records.append([round(N[i],2), round(P[i],2), round(K[i],2),
                        round(temp[i],2), round(hum[i],2), round(ph[i],2),
                        round(rain[i],2), crop])

df = pd.DataFrame(records, columns=['N','P','K','temperature','humidity','ph','rainfall','label'])
df.to_csv(os.path.join(DATA_DIR, 'Crop_recommendation.csv'), index=False)
print(f"Dataset: {df.shape[0]} rows, {df['label'].nunique()} crops")

# -----------------------------------------------------------
# 2. PREPROCESSING
# -----------------------------------------------------------
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# -----------------------------------------------------------
# 3. TRAIN ALL MODELS
# -----------------------------------------------------------
models = {
    'KNN':               KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':     DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, pred)
    results[name] = {'model': model, 'accuracy': acc, 'predictions': pred}
    print(f"{name}: {acc*100:.2f}%")

# K-Means (unsupervised — stored separately)
kmeans = KMeans(n_clusters=len(crop_profiles), random_state=42, n_init=10)
kmeans.fit(X_train_sc)

# -----------------------------------------------------------
# 4. SAVE BEST MODEL (Random Forest)
# -----------------------------------------------------------
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
with open(os.path.join(MODEL_DIR, 'kmeans.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)
print("Models saved.")

# -----------------------------------------------------------
# 5. GENERATE CHARTS
# -----------------------------------------------------------

# 5a. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(9, 5))
names  = list(results.keys())
accs   = [results[n]['accuracy']*100 for n in names]
colors = ['#4CAF50' if n == best_name else '#81C784' for n in names]
bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=1.5, zorder=3)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2E7D32')
ax.set_ylim(80, 102)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('ML Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax.yaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(STATIC_IMG_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5b. Feature Importance
fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors_fi = ['#1B5E20' if v == fi.max() else '#4CAF50' for v in fi.values]
fi.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold', pad=15)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(STATIC_IMG_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5c. Crop Distribution
crop_counts = df['label'].value_counts()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(crop_counts.index, crop_counts.values, color='#66BB6A', edgecolor='white', linewidth=1)
ax.set_xlabel('Crop', fontsize=11)
ax.set_ylabel('Sample Count', fontsize=11)
ax.set_title('Dataset — Samples per Crop', fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(crop_counts.index, rotation=45, ha='right', fontsize=9)
ax.yaxis.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(STATIC_IMG_DIR, 'crop_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5d. Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[['N','P','K','temperature','humidity','ph','rainfall']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greens', ax=ax,
            linewidths=0.5, linecolor='white', annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
fig.savefig(os.path.join(STATIC_IMG_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5e. Confusion Matrix (best model)
cm = confusion_matrix(y_test, results[best_name]['predictions'])
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.3, linecolor='white', annot_kws={'size': 7})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(STATIC_IMG_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5f. NPK Box Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
palette = {'N': '#4CAF50', 'P': '#FF9800', 'K': '#2196F3'}
for ax, nutrient in zip(axes, ['N', 'P', 'K']):
    top_crops = df.groupby('label')[nutrient].mean().nlargest(8).index
    data_plot = df[df['label'].isin(top_crops)]
    data_plot.boxplot(column=nutrient, by='label', ax=ax,
                      patch_artist=True,
                      boxprops=dict(facecolor=palette[nutrient], alpha=0.6),
                      medianprops=dict(color='white', linewidth=2))
    ax.set_title(f'{nutrient} by Crop (top 8)', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.spines[['top','right']].set_visible(False)
plt.suptitle('')
fig.suptitle('NPK Distribution Across Top Crops', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(STATIC_IMG_DIR, 'npk_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------
# 6. SAVE METRICS JSON
# -----------------------------------------------------------
metrics = {name: round(r['accuracy']*100, 2) for name, r in results.items()}
metrics['best_model'] = best_name
metrics['crop_classes'] = list(le.classes_)
with open(os.path.join(DATA_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)

# 7. Per-crop stats for research page
crop_stats = df.groupby('label')[['N','P','K','temperature','humidity','ph','rainfall']].mean().round(2)
crop_stats.to_json(os.path.join(DATA_DIR, 'crop_stats.json'), orient='index')

print("\nAll charts and metrics saved successfully.")
