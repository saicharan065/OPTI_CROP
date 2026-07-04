"""
OptiCrop — Flask Application
Routes: /, /predict, /suitability, /research, /api/predict
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import json
import os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

def load_artifacts():
    with open(os.path.join(BASE, 'model/crop_model.pkl'),   'rb') as f: model  = pickle.load(f)
    with open(os.path.join(BASE, 'model/scaler.pkl'),        'rb') as f: scaler = pickle.load(f)
    with open(os.path.join(BASE, 'model/label_encoder.pkl'), 'rb') as f: le     = pickle.load(f)
    with open(os.path.join(BASE, 'model/model_meta.json'))       as f: meta   = json.load(f)
    return model, scaler, le, meta

model, scaler, le, meta = load_artifacts()

CROP_ICONS = {
    'Rice':'🌾','Wheat':'🌾','Maize':'🌽','Chickpea':'🫘','KidneyBeans':'🫘',
    'PigeonPeas':'🫘','MothBeans':'🫘','MungBeans':'🫘','Blackgram':'🫘',
    'Lentil':'🫘','Pomegranate':'🍎','Banana':'🍌','Mango':'🥭','Grapes':'🍇',
    'Watermelon':'🍉','Muskmelon':'🍈','Apple':'🍎','Orange':'🍊','Papaya':'🍈',
    'Coconut':'🥥','Cotton':'🌿','Jute':'🌿','Coffee':'☕'
}

CROP_TIPS = {
    'Rice':        'Requires standing water; grow in lowland fields with good water retention.',
    'Wheat':       'Best sown in cool weather; ensure well-drained loamy soil.',
    'Maize':       'Plant in full sun; avoid waterlogging.',
    'Chickpea':    'Drought-tolerant legume; fix nitrogen naturally in soil.',
    'KidneyBeans': 'Needs warm soil; avoid over-watering to prevent root rot.',
    'PigeonPeas':  'Deep-rooted; excellent for dry semi-arid regions.',
    'MothBeans':   'Extremely drought-hardy; thrives in sandy soils.',
    'MungBeans':   'Short-duration crop; ideal for intercropping.',
    'Blackgram':   'Grows well in humid conditions; good protein source.',
    'Lentil':      'Cool-season crop; harvest before temperatures rise.',
    'Pomegranate': 'Frost-sensitive; prune regularly for best fruit yield.',
    'Banana':      'Heavy feeder; needs potassium-rich soil and consistent moisture.',
    'Mango':       'Deep taproot; minimal water once established.',
    'Grapes':      'Train on trellises; harvest in late summer.',
    'Watermelon':  'Needs hot days; plant after last frost in sandy loam.',
    'Muskmelon':   'High humidity tolerance; direct sun essential.',
    'Apple':       'Requires chill hours; cold winters improve fruit quality.',
    'Orange':      'Subtropical; protect from frost with windbreaks.',
    'Papaya':      'Fast-growing; harvest within 9 months of planting.',
    'Coconut':     'Coastal favourite; salt-tolerant and wind-resistant.',
    'Cotton':      'Long growing season; monitor for bollworms.',
    'Jute':        'Grown in flood-plains; harvested at flowering stage.',
    'Coffee':      'Shade-grown produces better flavour; avoid direct midday sun.',
}

def parse_inputs(form):
    return np.array([[
        float(form['nitrogen']),
        float(form['phosphorous']),
        float(form['potassium']),
        float(form['temperature']),
        float(form['humidity']),
        float(form['ph']),
        float(form['rainfall']),
    ]])

def predict_crop(inputs):
    scaled = scaler.transform(inputs)
    proba  = model.predict_proba(scaled)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(le.classes_[i], round(proba[i]*100, 1)) for i in top3_idx]
    return top3[0][0], top3[0][1], top3

def assess_suitability(crop, inputs):
    profiles = meta['crop_profiles']
    if crop not in profiles:
        return None, []
    p = profiles[crop]
    profile_keys = ['N','P','K','temp','hum','ph','rain']
    feat_names   = ['N','P','K','temperature','humidity','ph','rainfall']
    vals = inputs[0]
    issues, scores = [], []
    for i, (feat, pkey) in enumerate(zip(feat_names, profile_keys)):
        lo, hi = p[pkey]
        v = vals[i]
        if lo <= v <= hi:
            scores.append(1.0)
        else:
            gap     = min(abs(v-lo), abs(v-hi))
            rng     = hi - lo
            penalty = min(gap/(rng+1e-9), 1.0)
            scores.append(max(0, 1-penalty))
            direction = "low" if v < lo else "high"
            issues.append(f"{feat} is {direction} ({v:.1f}; ideal {lo}–{hi})")
    return round(np.mean(scores)*100, 1), issues

@app.route('/')
def index():
    return render_template('index.html', meta=meta)

@app.route('/predict', methods=['GET','POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            inputs = parse_inputs(request.form)
            best_crop, confidence, top3 = predict_crop(inputs)
            result = {
                'crop':       best_crop,
                'confidence': confidence,
                'icon':       CROP_ICONS.get(best_crop, '🌱'),
                'tip':        CROP_TIPS.get(best_crop, ''),
                'top3':       top3,
                'inputs':     {k: float(request.form[k]) for k in
                               ['nitrogen','phosphorous','potassium',
                                'temperature','humidity','ph','rainfall']},
            }
        except Exception as e:
            result = {'error': str(e)}
    return render_template('predict.html', result=result, meta=meta)

@app.route('/suitability', methods=['GET','POST'])
def suitability():
    result = None
    if request.method == 'POST':
        try:
            inputs = parse_inputs(request.form)
            crop   = request.form['crop']
            suit, issues = assess_suitability(crop, inputs)
            level = ('Excellent' if suit>=80 else
                     'Good'      if suit>=60 else
                     'Moderate'  if suit>=40 else 'Poor')
            result = {
                'crop':        crop,
                'suitability': suit,
                'level':       level,
                'issues':      issues,
                'icon':        CROP_ICONS.get(crop,'🌱'),
                'tip':         CROP_TIPS.get(crop,''),
            }
        except Exception as e:
            result = {'error': str(e)}
    return render_template('suitability.html',
                           result=result, crops=meta['crops'], meta=meta)

@app.route('/research')
def research():
    return render_template('research.html', meta=meta)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    try:
        inputs = np.array([[data['N'],data['P'],data['K'],
                            data['temperature'],data['humidity'],
                            data['ph'],data['rainfall']]])
        crop, confidence, top3 = predict_crop(inputs)
        return jsonify({'crop':crop,'confidence':confidence,
                        'top3':[{'crop':c,'probability':p} for c,p in top3]})
    except Exception as e:
        return jsonify({'error':str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
