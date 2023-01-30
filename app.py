#API
# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI, Path
from PreClass import PreClass
import numpy as np
import pickle
import pandas as pd 

# 2. Create the app object
app = FastAPI()

# Chargement du modèle sauvegardé avec pickle
try:
    pickle_in = open("clf.pkl", "rb")
    clf = pickle.load(pickle_in)
except:
    print("Erreur lors du chargement du modèle sauvegardé")

# Chargement des données
try :
    path = "/Users/adpro/Desktop/Scoring_Model/data_api.csv"
    data = pd.read_csv(path)
except:
    print("Erreur lors du chargemet des données")


app = FastAPI()
@app.get("/predict/{customer_id}")
def predict(customer_id: int = Path(..., gt=0)):
    # Récupération des données du client
    customer_data = data[data["ID"] == customer_id]
    if customer_data.empty:
        return {"error": "Aucun client trouvé avec cet ID"}
    
    # Sélection des variables à utiliser pour la prédiction
    feats = customer_data.drop(["ID", "Target"], axis=1)
    
    #probabilité
    try:
        proba = dict(clf.predict_proba([feats][0]).tolist())
        for key in proba.keys():
            proba = proba[key]
    except: 
        print("erreur lors du calcul de probabilité")
    
    
    return {"probabilite":proba}
    
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload