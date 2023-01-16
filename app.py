#API
# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
from PreClass import PreClass
import numpy as np
import pickle
import pandas as pd 

# 2. Create the app object
app = FastAPI()
pickle_in = open("clf.pkl","rb")
clf = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def prep_p(data:PreClass):
    data = data.dict()
    features = [data[f] for f in ["EXT_SOURCE_3", "EXT_SOURCE_2", "CODE_GENDER", "DAYS_REGISTRATION", "DAYS_BIRTH", "PAYMENT_RATE"]]
    prediction = clf.predict([features])[0]
    prob_pred = clf.predict_proba([features])
    prob_pred_format =  dict(prob_pred.tolist())
    for key in prob_pred_format.keys():
        proba_value_1 = prob_pred_format[key]
        
    prediction = "Client solvable" if prediction > 0.5 else "Client non solvable"
    return {
        'prediction': prediction,
        'probabilité': round(proba_value_1*100,2)
    }
    
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload