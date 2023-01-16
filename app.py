API
# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
from PreClass import PreClass
from fproba import prediprob
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
    EXT_SOURCE_3 = data["EXT_SOURCE_3"]
    EXT_SOURCE_2 = data["EXT_SOURCE_2"]
    CODE_GENDER = data["CODE_GENDER"]
    DAYS_REGISTRATION = data["DAYS_REGISTRATION"] 
    DAYS_BIRTH = data["DAYS_BIRTH"]
    PAYMENT_RATE = data["PAYMENT_RATE"]
     # prediction de la classe
    prediction = clf.predict([[EXT_SOURCE_3,EXT_SOURCE_2,CODE_GENDER,DAYS_REGISTRATION,DAYS_BIRTH,PAYMENT_RATE]])
     # proba que le client soit solvable
    prob_pred = clf.predict_proba([[EXT_SOURCE_3,EXT_SOURCE_2,CODE_GENDER,DAYS_REGISTRATION,DAYS_BIRTH,PAYMENT_RATE]])
    prob_pred_format = dict(prob_pred.tolist())
    for key in prob_pred_format.keys():
        proba_value_1 = prob_pred_format[key]
        
    
    if(prediction[0]>0.5):
        prediction="Client solvable"
    else:
        prediction="Client non solvable"
        
    return {
        'prediction': prediction,
        'prob_pred': prob_pred,
        'format' : prob_pred_format,
        'probabilité': proba_value_1
    }
    
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload