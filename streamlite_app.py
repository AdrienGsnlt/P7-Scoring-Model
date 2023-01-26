import streamlit as st
import pandas as pd 
import plost
import pickle
import sklearn
#Classifier
import lightgbm
from lightgbm import LGBMClassifier
### Check Accuracy
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt




#Chargement des données

@st.cache
def load_data(path):
    data = pd.read_csv(path)
    data = data.set_index("ID") 
    return data

@st.cache
def id_client(data,id):
    data_client = data[data.index == int(id)]
    return data_client

@st.cache
def prediction_client(model,data,id):
    data_client = id_client(data,id)
    prediction = model.predict(data_client)
    prediction = "Client solvable" if prediction > 0.5 else "Client non solvable"
    prob_pred = model.predict_proba(data_client)[:,1]
    prob_pred = round(prob_pred.item()*100,2)
    return prediction, prob_pred

#importation des données
path = "/Users/adpro/Desktop/Scoring_Model/data_api.csv"
data = load_data(path)

#preparation prediction
X=data.iloc[:,:-1]
y= data["Target"]

#chargement du modèle
pickle_in = open("clf.pkl","rb")
clf = pickle.load(pickle_in)



########### Main Content

st.title("Dashboard")
st.header("Solvabilité du client")
st.subheader("Description du client")


##Input client
number_id = st.number_input("ID du client", min_value = 100002,max_value=134827)

if st.checkbox("Voir détails informations client"):
    st.write("**Le client actuel est  :**", number_id)
    ##Description du client 
    desc_client = id_client(X,number_id)
    st.write("**Age : {:.0f} ans**".format(int(-desc_client["DAYS_BIRTH"]/365)))
    st.write("**EXT_SOURCE_2 :**", desc_client["EXT_SOURCE_2"].values[0])
    st.write("**EXT_SOURCE_3 :**", desc_client["EXT_SOURCE_3"].values[0])
    st.write("**DAYS_REGISTRATION :**", desc_client["DAYS_REGISTRATION"].values[0])
    st.write("**DAYS_ID_PUBLISH :**", desc_client["DAYS_ID_PUBLISH"].values[0])
    st.write("**DAYS_LAST_PHONE_CHANGE :**", desc_client["DAYS_LAST_PHONE_CHANGE"].values[0])
    st.write("**PAYMENT_RATE :**", desc_client["PAYMENT_RATE"].values[0])
    st.write("**REGION_POPULATION_RELATIVE :**", desc_client["REGION_POPULATION_RELATIVE"].values[0])
        

### Solvabilite du client
st.subheader("Solvabilité du client")

b = prediction_client(clf,X,number_id)
b[0]
b[1]

st.header("Description de la population")

### Population 
option_feat= st.sidebar.selectbox("Features", ("Age","Ext2","Ext3","Publication","Registration","Phone","Payment_Rate","Region_Pop"))
st.header(option_feat)

if option_feat == "Age":
    st.subheader("boxplot age")
    ages = data["DAYS_BIRTH"]/-365
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=ages, color = "lightblue")
    #higlight
    ax.axvline(ages[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)


if option_feat == "Ext2":
    #EXT_SOURCE_2
    st.subheader("boxplot EXT_SOURCE_2")
    ext2_pop = data["EXT_SOURCE_2"]
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=ext2_pop, color = "lightblue")
    #higlight
    ax.axvline(ext2_pop[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Ext3":
    #EXT_SOURCE_3
    st.subheader("boxplot EXT_SOURCE_3")
    ext3_pop = data["EXT_SOURCE_3"]
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=ext3_pop, color = "lightblue")
    #higlight
    ax.axvline(ext3_pop[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Registration":
    #DAYS_REGISTRATION
    st.subheader("boxplot DAYS_REGISTRATION")
    days_regis = data["DAYS_REGISTRATION"]/-365
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=days_regis, color = "lightblue")
    #higlight
    ax.axvline(days_regis[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Publication":
    #DAYS_ID_PUBLISH
    st.subheader("boxplot DAYS_ID_PUBLISH")
    days_publish = data["DAYS_ID_PUBLISH"]/-365
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=days_publish, color = "lightblue")
    #higlight
    ax.axvline(days_publish[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Phone": 
    #DAYS_LAST_PHONE_CHANGE
    st.subheader("boxplot DAYS_LAST_PHONE_CHANGE")
    days_phone = data["DAYS_LAST_PHONE_CHANGE"]/-365
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=days_phone, color = "lightblue")
    #higlight
    ax.axvline(days_phone[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Payment_Rate":  
    #PAYMENT_RATE
    st.subheader("boxplot PAYMENT_RATE")
    payment = data["PAYMENT_RATE"]
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=payment, color = "lightblue")
    #higlight
    ax.axvline(payment[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Region_Pop":
    #REGION_POPULATION_RELATIVE
    st.subheader("boxplot REGION_POPULATION_RELATIVE")
    region_pop = data["REGION_POPULATION_RELATIVE"]
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=region_pop, color = "lightblue")
    #higlight
    ax.axvline(region_pop[number_id],color="red",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)

