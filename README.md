🌾 OptiCrop — Smart Agricultural Production Optimization Engine

OptiCrop is a full-stack machine learning web application that helps recommend the most suitable crop to grow based on soil and environmental conditions. It combines a trained ML model with a clean, responsive web interface to make data-driven crop recommendations accessible to farmers, students, and researchers alike.

🔗 Live Demo: opti-crop-blue.vercel.app


📌 Overview

Agricultural productivity depends heavily on choosing the right crop for the right conditions. OptiCrop uses a machine learning model trained on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall data to predict the most optimal crop for a given set of conditions — helping reduce guesswork and improve yield outcomes.


🚀 Tech Stack

LayerTechnologyBackendFlask (Python)Machine LearningScikit-learn — Random Forest Classifier (~92% accuracy)FrontendHTML, Bootstrap 5, Jinja2 templatingDeploymentVercel


🧠 How It Works


User inputs soil and climate parameters (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall).
The Flask backend processes the input and feeds it to the trained Random Forest model.
The model predicts the most suitable crop from the dataset's crop classes.
The result is rendered back to the user through a Jinja2-templated, Bootstrap-styled interface.



📁 Project Structure

This repository is organized by project lifecycle phase, following an academic project workflow:

OPTI_CROP/
├── 1. Brainstorming & Ideation.
├── 2. Requirement Analysis.
├── 3. Project Design Phase.
├── 4. Project Planning Phase.
├── 5. Project Development Phase        # Core Flask app, ML model, templates
├── 6. Project Testing.
├── 7. Project Documentation.
└── 8. Project Demonstration.           # Demo videos, walkthroughs


⚙️ Getting Started

Prerequisites


Python 3.8+
pip


Installation

bash# Clone the repository
git clone https://github.com/saicharan065/OPTI_CROP.git
cd OPTI_CROP

# Navigate to the development phase folder
cd "5. Project Development Phase"

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

The app should now be running locally at http://127.0.0.1:5000.


📊 Model Performance

The crop recommendation model is a Random Forest Classifier trained using Scikit-learn, achieving approximately 92% accuracy on the test dataset.


🎥 Demonstration

Demo materials, including walkthrough videos and documentation, are available in the 8. Project Demonstration. folder.


👤 Author

Sai Charan Samudrala
B.Tech Computer Science, SRM University AP




📄 License

This project was developed for academic purposes as part of a college project submission.
