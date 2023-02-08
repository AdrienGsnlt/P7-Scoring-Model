# Library imports
import gunicorn
from flask import Flask, request, jsonify
import numpy as np
import pickle5 as pickle
import pandas as pd 
from flask import Flask, jsonify

app = Flask(__name__)

try:
    pickle_in = open("clf.pkl", "rb")
    clf = pickle.load(pickle_in)
except:
    print("Error loading the saved model")

try:
    path = "https://github.com/AdrienGsnlt/P7-Scoring-Model/blob/master/data_api.csv"
    data = pd.read_csv(path)
except:
    print("Error loading the data")

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.route("/predict/<int:customer_id>", methods=["POST"])
def predict(customer_id):
    # Récupération des données du client
    customer_data = data[data["ID"] == customer_id]
    if customer_data.empty:
        return jsonify({"error": "Aucun client trouvé avec cet ID"})
    
    # Sélection des variables à utiliser pour la prédiction
    feats = customer_data.drop(["ID", "Target"], axis=1)
    
    #probabilité
    try:
        proba = dict(clf.predict_proba([feats][0]).tolist())
        for key in proba.keys():
            proba = proba[key]
    except: 
        print("erreur lors du calcul de probabilité")
    
    return jsonify({"probabilite":proba})

if __name__ == "__main__":
    app.run(debug=True)
