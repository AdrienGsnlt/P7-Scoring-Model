#####
# Importation des librairies
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
import plotly.express as px
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler


###########
#definition des fonctions utilies

#@st.cache
def load_data(path):
    data = pd.read_csv(path) 
    return data

#@st.cache
def id_client(data,id):
    data_client = data[data["ID"] == number_id]
    data_client = data_client.drop(["ID", "Target"], axis=1)
    return data_client

## Jauge colorée
def color_jauge(score):
    if score >= 95:
        return "rgb(0, 255, 0)"
    elif score >= 90:
        r = 51 * (95 - score) 
        g = 255
        return "rgb({}, {}, 0)".format(r, g)
    elif score >= 85:
        r = 255 
        g = 255 - 51 * (90 - score) 
        return "rgb({}, {}, 0)".format(r, g)
    else:
        return "rgb(255, 0, 0)"

########
# Importation des données
path = "/Users/adpro/Desktop/Scoring_Model/data_api.csv"
data = load_data(path)




#####################################
# Main Content
#####################################

st.title("Score de solvabilité banquaire")
st.header("Evaluation du client")

##Input client
number_id = st.sidebar.number_input("ID du client",min_value = min(data["ID"]),max_value=max(data["ID"]))

#processing des données
client_data = id_client(data,number_id)

### Solvabilite du client


url = "http://127.0.0.1:5000/predict/{customer_id}"

response = requests.get(url.format(customer_id=number_id))
result = response.json()
prediction = round(result['probabilite'],4)

st.write("La probabilité que le client", number_id, "soit solvable est de :", prediction*100,"%")

# Jauge colorée
score = prediction*100
color = color_jauge(score)
fig = px.pie(values=[score+1], hole=0.01, color_discrete_sequence=[color], height=400, width=400)
fig.update_traces(textinfo='none')
st.plotly_chart(fig)

##Description du client
if st.sidebar.checkbox("Voir détails informations client"): 
    st.write("**GENDER**".client_data["GENDER"].values[0])
    st.write("**BUSINESS_TYPE :**", client_data["BUSINESS_TYPE"].values[0])
    st.write("**EXT3 :**", client_data["EXT3"].values[0])
    st.write("**REGION_RATING :**", client_data["REGION_RATING"].values[0])
    st.write("**UNACCOMPANIED :**", client_data["UNACCOMPANIED"].values[0])
    st.write("**EXT2 :**", client_data["EXT2"].values[0])
    st.write("**INCOME_TYPE :**", client_data["INCOME_TYPE"].values[0])
        

st.header("Comparaison du client à la base de données")
st.subheader("Profil du client vs clients solvables vs non solvables")

# Radar Plot
## scaling des data du radar plot
scaler = StandardScaler()

st_data = scaler.fit_transform(data.drop(['ID','Target'],axis=1))
st_data = pd.DataFrame(st_data, columns=data.drop(['ID','Target'],axis=1).columns)
st_data["Target"] = data["Target"]
st_data["ID"] = data["ID"]

## création d'une dataframe avec les données du clients, le profil moyen des clients solvables, et celui des clients non solvables

### MODIF COULEUR XX
client_data_0 = st_data[st_data['ID'] == number_id].drop(['ID', 'Target'], axis=1)
mean_data_0 = st_data[st_data['Target'] == 0].drop(['ID', 'Target'], axis=1).mean().to_frame().T
mean_data_1 = st_data[st_data['Target'] == 1].drop(['ID', 'Target'], axis=1).mean().to_frame().T
scaled_df = pd.concat([client_data_0, mean_data_0, mean_data_1], ignore_index=True)
scaled_df['Target'] = ['Client'] + ['Non_Solvable'] * len(mean_data_0) + ['Solvable'] * len(mean_data_1)

## Création du radar plot
number_of_variables = len(scaled_df.columns) - 1
theta = np.linspace(0, 2 * np.pi, number_of_variables, endpoint=False)
theta = np.concatenate([theta, [theta[0]]])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)
ax.set_xticks(theta[:-1])
ax.set_xticklabels(scaled_df.columns[:-1])

colors = {"Client": "blue", "Solvable": "green", "Non_Solvable": "red"}
for i in range(len(scaled_df)):
    values = scaled_df.iloc[i].drop('Target').values.flatten().tolist()
    values += values[:1]
    ax.plot(theta, values, 'o-', linewidth=2, color=colors[scaled_df.iloc[i]["Target"]])

ax.legend(scaled_df['Target'].unique())
st.pyplot(fig)


## Détails sur chaque features
st.subheader("Details sur chaque features")
option_feat= st.sidebar.selectbox("Features", ("GENDER","BUSINESS_TYPE","EXT3","REGION_RATING","UNACCOMPANIED","EXT2","INCOME_TYPE"))
st.header(option_feat)

if option_feat == "GENDER":
    st.subheader("boxplot GENDER")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data["GENDER"], color = "lightblue")
    #higlight
    ax.axvline((data.loc[data["Target"]==1,"GENDER"].mean()),color="green",linestyle='--')
    ax.axvline((data.loc[data["Target"]==0,"GENDER"].mean()),color="red",linestyle='--')
    ax.axvline((client_data["GENDER"].values[0]),color="blue",linestyle='--')
    ax.set(title='Box plot Genre des clients',xlabel="Genre",ylabel ='')
    st.pyplot(fig)


if option_feat == "BUSINESS_TYPE":
    #EXT_SOURCE_2
    st.subheader("boxplot BUSINESS_TYPE")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data["BUSINESS_TYPE"], color = "lightblue")
    #higlight
    ax.axvline((data.loc[data["Target"]==1,"BUSINESS_TYPE"].mean()),color="green",linestyle='--')
    ax.axvline((data.loc[data["Target"]==0,"BUSINESS_TYPE"].mean()),color="red",linestyle='--')
    ax.axvline(client_data["BUSINESS_TYPE"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot BUSINESS_TYPE',xlabel="BUSINESS_TYPE",ylabel ='')
    st.pyplot(fig)

if option_feat == "EXT3":
    #EXT_SOURCE_3
    st.subheader("boxplot EXT3")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data["EXT3"], color = "lightblue")
    #higlight
    ax.axvline((data.loc[data["Target"]==1,"EXT3"].mean()),color="green",linestyle='--')
    ax.axvline((data.loc[data["Target"]==0,"EXT3"].mean()),color="red",linestyle='--')
    ax.axvline(client_data["EXT3"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot EXT3',xlabel="EXT3",ylabel ='')
    st.pyplot(fig)


if option_feat == "REGION_RATING":
    #DAYS_REGISTRATION
    st.subheader("boxplot REGION_RATING")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data["REGION_RATING"], color = "lightblue")
    #higlight
    ax.axvline((data.loc[data["Target"]==1,"REGION_RATING"].mean()),color="green",linestyle='--')
    ax.axvline((data.loc[data["Target"]==0,"REGION_RATING"].mean()),color="red",linestyle='--')
    ax.axvline(client_data["REGION_RATING"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot REGION_RATING',xlabel="REGION_RATING",ylabel ='')
    st.pyplot(fig)

if option_feat == "UNACCOMPANIED":
    #DAYS_ID_PUBLISH
    st.subheader("boxplot UNACCOMPANIED")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data["UNACCOMPANIED"], color = "lightblue")
    #higlight
    ax.axvline((data.loc[data["Target"]==1,"UNACCOMPANIED"].mean()),color="green",linestyle='--')
    ax.axvline((data.loc[data["Target"]==0,"UNACCOMPANIED"].mean()),color="red",linestyle='--')
    ax.axvline(client_data["UNACCOMPANIED"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot date publish',xlabel="Temps (year)",ylabel ='')
    st.pyplot(fig)

if option_feat == "EXT2": 
    #DAYS_LAST_PHONE_CHANGE
    st.subheader("boxplot EXT2")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data["EXT2"], color = "lightblue")
    #higlight
    ax.axvline((data.loc[data["Target"]==1,"EXT2"].mean()),color="green",linestyle='--')
    ax.axvline((data.loc[data["Target"]==0,"EXT2"].mean()),color="red",linestyle='--')
    ax.axvline(client_data["EXT2"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot EXT2',xlabel="EXT2",ylabel ='')
    st.pyplot(fig)

if option_feat == "INCOME_TYPE":  
    #PAYMENT_RATE
    st.subheader("boxplot INCOME_TYPE")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data["INCOME_TYPE"], color = "lightblue")
    #higlight
    ax.axvline((data.loc[data["Target"]==1,"INCOME_TYPE"].mean()),color="green",linestyle='--')
    ax.axvline((data.loc[data["Target"]==0,"INCOME_TYPE"].mean()),color="red",linestyle='--')
    ax.axvline(client_data["INCOME_TYPE"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot INCOME_TYPE',xlabel="INCOME_TYPE",ylabel ='')
    st.pyplot(fig)


### Scatter plot
st.header("Scatter Plot")
color_client = color_jauge(score)
fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(data=data, x=round(data["GENDER"],2),y=data["BUSINESS_TYPE"],hue="Target")
sns.scatterplot(data=client_data, x=round(client_data["GENDER"],2),y=client_data["BUSINESS_TYPE"], color='red', s=400)
st.pyplot(fig)