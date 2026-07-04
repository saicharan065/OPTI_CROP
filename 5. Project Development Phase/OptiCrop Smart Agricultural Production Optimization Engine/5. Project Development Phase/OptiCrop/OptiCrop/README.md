# 🌿 OptiCrop — Smart Agricultural Production Optimization Engine

An AI-powered crop recommendation system built with **Flask + Scikit-learn** that analyzes soil nutrients and environmental conditions to deliver intelligent crop recommendations.

---

## 🚀 Quick Start

```bash
# 1. Clone / navigate to project
cd OptiCrop

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate         # Linux/Mac

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Train ML models (first time only)
python model/train_models.py
# or use generate_and_train.py for extended dataset and charts
python generate_and_train.py

# 5. Run Flask app
python app.py
# → Open http://localhost:5000
```

---

## 📁 Project Structure

```
OptiCrop/
├── app.py                    # Flask application (all routes)
├── requirements.txt
├── README.md
├── data/
│   └── crop_data.csv         # Generated training dataset (2,760 records, 23 crops)
├── model/
│   ├── train_models.py       # ML training pipeline
│   ├── crop_model.pkl        # Best model (Random Forest, 92.57%)
│   ├── scaler.pkl            # StandardScaler
│   ├── label_encoder.pkl     # LabelEncoder (23 crops)
│   ├── kmeans.pkl            # K-Means clustering model
│   └── model_meta.json       # Metrics, feature importances, crop profiles
└── templates/
    ├── base.html             # Shared layout & navbar
    ├── index.html            # Home page (Scenario overview)
    ├── predict.html          # Scenario 1 – Crop Recommendation
    ├── suitability.html      # Scenario 2 – Suitability Assessment
    └── research.html         # Scenario 3 – Research & Analytics
```

---

## 🤖 Machine Learning Models

| Algorithm            | Accuracy | F1 Score | CV Score |
|----------------------|----------|----------|----------|
| K-Nearest Neighbors  | 85.87%   | 85.68%   | 85.87%   |
| Logistic Regression  | 84.96%   | 84.78%   | 87.41%   |
| Decision Tree        | 89.31%   | 89.34%   | 88.27%   |
| **Random Forest** ✅  | **92.57%** | **92.48%** | **91.85%** |

> **Random Forest** selected as the production model.

---

## 🌾 Supported Crops (23)

Rice, Wheat, Maize, Chickpea, KidneyBeans, PigeonPeas, MothBeans, MungBeans,
Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon,
Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 📡 REST API

**POST** `/api/predict`  
```json
{
  "N": 80, "P": 45, "K": 45,
  "temperature": 25, "humidity": 80,
  "ph": 6.5, "rainfall": 200
}
```
**Response:**
```json
{
  "crop": "Rice",
  "confidence": 91.4,
  "top3": [
    {"crop": "Rice", "probability": 91.4},
    {"crop": "Jute", "probability": 5.2},
    {"crop": "Maize", "probability": 3.4}
  ]
}
```

---

## 🔬 Three Scenarios

| Scenario | Route | Description |
|----------|-------|-------------|
| 1 — Crop Recommendation | `/predict` | Best crop for given soil/climate conditions |
| 2 — Suitability Assessment | `/suitability` | Compatibility score for a chosen crop |
| 3 — Research & Analytics | `/research` | Model metrics, feature importances, crop profiles |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, Flask
- **ML**: Scikit-learn (Random Forest, KNN, Logistic Regression, Decision Tree, K-Means)
- **Data**: Pandas, NumPy
- **Visualization**: Chart.js (frontend), Matplotlib/Seaborn (analysis)
- **Frontend**: HTML5, Bootstrap 5.3, Bootstrap Icons
- **Persistence**: Pickle (.pkl)
- **Version Control**: Git

---

## 👨‍💻 Author

Sai Charan — SRM University AP  
GitHub: [github.com/saicharan065](https://github.com/saicharan065)
