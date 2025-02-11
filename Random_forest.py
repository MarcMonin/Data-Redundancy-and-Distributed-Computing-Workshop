import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger le dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# Entraîner un modèle
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, "random_forest_model.pkl")

# Charger le modèle
model = joblib.load("random_forest_model.pkl")

# Initialiser Flask
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Récupérer les paramètres de la requête
        features = [float(request.args.get(f'feature{i}', 0)) for i in range(30)]
        
        # Faire une prédiction
        prediction = model.predict([features])[0]
        
        # Retourner le résultat
        return jsonify({"model": "random_forest", "prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5004, debug=True)
