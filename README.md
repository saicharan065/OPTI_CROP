# 🌾 OptiCrop — Smart Agricultural Production Optimization Engine

OptiCrop is a full-stack machine learning web application that helps recommend the most suitable crop to grow based on soil and environmental conditions. It combines a trained ML model with a clean, responsive web interface to make data-driven crop recommendations accessible to farmers, students, and researchers alike.

🔗 **Live Demo:** [opti-crop-21mj28erl-s-sai-charans-projects.vercel.app](https://opti-crop-git-main-s-sai-charans-projects.vercel.app/)

---

## 📌 Overview

Agricultural productivity depends heavily on choosing the right crop for the right conditions. OptiCrop uses a machine learning model trained on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall data to predict the most optimal crop for a given set of conditions — helping reduce guesswork and improve yield outcomes.

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| Machine Learning | Scikit-learn — Random Forest Classifier (~92% accuracy) |
| Frontend | HTML, Bootstrap 5, Jinja2 templating |
| Deployment | Vercel |

---

## 🧠 How It Works

1. User inputs soil and climate parameters (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall).
2. The Flask backend processes the input and feeds it to the trained Random Forest model.
3. The model predicts the most suitable crop from the dataset's crop classes.
4. The result is rendered back to the user through a Jinja2-templated, Bootstrap-styled interface.

---

## 📁 Project Structure

This repository is organized by project lifecycle phase, following an academic project workflow:

```
OPTI_CROP/
├── 1. Brainstorming & Ideation/
│   ├── Brainstorming & Idea Prioritization.pdf
│   ├── Define Problem Statements.pdf
│   └── Empathy Map.pdf
│
├── 2. Requirement Analysis/
│   ├── Customer Journey Map.pdf
│   ├── Data Flow Diagram.pdf
│   ├── Solution Requirements.pdf
│   └── Technology Stack.pdf
│
├── 3. Project Design Phase/
│   ├── Problem-Solution Fit.pdf
│   ├── Proposed Solution.pdf
│   └── Solution Architecture.pdf
│
├── 4. Project Planning Phase/
│   └── Project Planning.pdf
│
├── 5. Project Development Phase/        # Core Flask app, ML model, templates
│   ├── OptiCrop Smart Agricultural Pro.../   # Application source code
│   ├── Code-Layout, Readability and R....pdf
│   ├── Coding & Solution.pdf
│   ├── No. of Functional Features Inclu....pdf
│   └── README.md
│
├── 6. Project Testing/
│   └── Performance Testing.pdf
│
├── 7. Project Documentation/
│   ├── Project Executable Files.pdf
│   └── Sample Project Documentation....pdf
│
└── 8. Project Demonstration/           # Demo videos, walkthroughs
    ├── Communication.pdf
    ├── Demonstration of Proposed Fea....pdf
    ├── Project Demo Planning.pdf
    ├── Scalability & Future Plan.pdf
    ├── Team Involvement in Demonstr....pdf
    └── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/saicharan065/OPTI_CROP.git
cd OPTI_CROP

# Navigate to the application source folder
cd "5. Project Development Phase/OptiCrop Smart Agricultural Pro..."

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app should now be running locally at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 📊 Model Performance

The crop recommendation model is a Random Forest Classifier trained using Scikit-learn, achieving approximately 92% accuracy on the test dataset.

---

## 🎥 Demonstration

Demo materials, including walkthrough videos, communication docs, and scalability/future plans, are available in the `8. Project Demonstration/` folder.

---

## 👤 Author

**Sai Charan**
B.Tech Computer Science, SRM University AP
