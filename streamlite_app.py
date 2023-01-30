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

def proc_date(data):
    data_proc = data.copy()
    data_proc["DAYS_BIRTH"] = -data_proc["DAYS_BIRTH"]/365
    data_proc["DAYS_REGISTRATION"] = -data_proc["DAYS_REGISTRATION"]/365
    data_proc["DAYS_ID_PUBLISH"] = -data_proc["DAYS_ID_PUBLISH"]/365
    data_proc["DAYS_LAST_PHONE_CHANGE"] = -data_proc["DAYS_LAST_PHONE_CHANGE"]/365
    return data_proc

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
data_proc = proc_date(data)
client_data = id_client(data,number_id)
client_data_proc = proc_date(client_data)

### Solvabilite du client


url = "http://127.0.0.1:8000/predict/{customer_id}"

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
    st.write("**Age : {:.0f} ans**".format(int(client_data_proc["DAYS_BIRTH"])))
    st.write("**EXT_SOURCE_2 :**", client_data_proc["EXT_SOURCE_2"].values[0])
    st.write("**EXT_SOURCE_3 :**", client_data_proc["EXT_SOURCE_3"].values[0])
    st.write("**DAYS_REGISTRATION :**", client_data_proc["DAYS_REGISTRATION"].values[0])
    st.write("**DAYS_ID_PUBLISH :**", client_data_proc["DAYS_ID_PUBLISH"].values[0])
    st.write("**DAYS_LAST_PHONE_CHANGE :**", client_data_proc["DAYS_LAST_PHONE_CHANGE"].values[0])
    st.write("**PAYMENT_RATE :**", client_data_proc["PAYMENT_RATE"].values[0])
    st.write("**REGION_POPULATION_RELATIVE :**", client_data_proc["REGION_POPULATION_RELATIVE"].values[0])
        

st.header("Domparaison du client à la base de données")
st.subheader("Profil du client vs clients solvables vs non solvables")

# Radar Plot
## scaling des data du radar plot
scaler = StandardScaler()

st_data = scaler.fit_transform(data_proc.drop(['ID','Target'],axis=1))
st_data = pd.DataFrame(st_data, columns=data_proc.drop(['ID','Target'],axis=1).columns)
st_data["Target"] = data_proc["Target"]
st_data["ID"] = data_proc["ID"]

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
option_feat= st.sidebar.selectbox("Features", ("Age","Ext2","Ext3","Publication","Registration","Phone","Payment_Rate","Region_Pop"))
st.header(option_feat)

if option_feat == "Age":
    st.subheader("boxplot age")
    ages = data_proc["DAYS_BIRTH"]
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=ages, color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"DAYS_BIRTH"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"DAYS_BIRTH"].mean()),color="red",linestyle='--')
    ax.axvline((client_data_proc["DAYS_BIRTH"].values[0]),color="blue",linestyle='--')
    ax.set(title='Box plot age des clients',xlabel="Age (ans)",ylabel ='')
    st.pyplot(fig)


if option_feat == "Ext2":
    #EXT_SOURCE_2
    st.subheader("boxplot EXT_SOURCE_2")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data_proc["EXT_SOURCE_2"], color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"EXT_SOURCE_2"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"EXT_SOURCE_2"].mean()),color="red",linestyle='--')
    ax.axvline(client_data_proc["EXT_SOURCE_2"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot Ext Source 2',xlabel="Ext_source 2)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Ext3":
    #EXT_SOURCE_3
    st.subheader("boxplot EXT_SOURCE_3")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data_proc["EXT_SOURCE_3"], color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"EXT_SOURCE_3"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"EXT_SOURCE_3"].mean()),color="red",linestyle='--')
    ax.axvline(client_data_proc["EXT_SOURCE_3"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot EXT Source 3',xlabel="Ext_source 3",ylabel ='')
    st.pyplot(fig)

if option_feat == "Registration":
    #DAYS_REGISTRATION
    st.subheader("boxplot DAYS_REGISTRATION")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data_proc["DAYS_REGISTRATION"], color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"DAYS_REGISTRATION"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"DAYS_REGISTRATION"].mean()),color="red",linestyle='--')
    ax.axvline(client_data_proc["DAYS_REGISTRATION"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot date Registration',xlabel="Temps (year)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Publication":
    #DAYS_ID_PUBLISH
    st.subheader("boxplot DAYS_ID_PUBLISH")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data_proc["DAYS_ID_PUBLISH"], color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"DAYS_ID_PUBLISH"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"DAYS_ID_PUBLISH"].mean()),color="red",linestyle='--')
    ax.axvline(client_data_proc["DAYS_ID_PUBLISH"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot date publish',xlabel="Temps (year)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Phone": 
    #DAYS_LAST_PHONE_CHANGE
    st.subheader("boxplot DAYS_LAST_PHONE_CHANGE")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data_proc["DAYS_LAST_PHONE_CHANGE"], color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"DAYS_LAST_PHONE_CHANGE"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"DAYS_LAST_PHONE_CHANGE"].mean()),color="red",linestyle='--')
    ax.axvline(client_data_proc["DAYS_LAST_PHONE_CHANGE"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot date last phone change',xlabel="Temps (year)",ylabel ='')
    st.pyplot(fig)

if option_feat == "Payment_Rate":  
    #PAYMENT_RATE
    st.subheader("boxplot PAYMENT_RATE")
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data_proc["PAYMENT_RATE"], color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"PAYMENT_RATE"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"PAYMENT_RATE"].mean()),color="red",linestyle='--')
    ax.axvline(client_data_proc["PAYMENT_RATE"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot payment rate',xlabel="Payment Rate",ylabel ='')
    st.pyplot(fig)

if option_feat == "Region_Pop":
    #REGION_POPULATION_RELATIVE
    st.subheader("boxplot REGION_POPULATION_RELATIVE")
    region_pop = data_proc["REGION_POPULATION_RELATIVE"]
    #box object
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x=data_proc["REGION_POPULATION_RELATIVE"], color = "lightblue")
    #higlight
    ax.axvline((data_proc.loc[data_proc["Target"]==1,"REGION_POPULATION_RELATIVE"].mean()),color="green",linestyle='--')
    ax.axvline((data_proc.loc[data_proc["Target"]==0,"REGION_POPULATION_RELATIVE"].mean()),color="red",linestyle='--')
    ax.axvline(client_data_proc["REGION_POPULATION_RELATIVE"].values[0],color="blue",linestyle='--')
    ax.set(title='Box plot region population relative',xlabel="Region Pop",ylabel ='')
    st.pyplot(fig)


### Scatter plot
st.header("Scatter Plot")
color_client = color_jauge(score)
fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(data=data, x=round(data["EXT_SOURCE_2"],2),y=data["DAYS_REGISTRATION"]/-365,hue="Target")
sns.scatterplot(data=client_data_proc, x=round(client_data_proc["EXT_SOURCE_2"],2),y=client_data_proc["DAYS_REGISTRATION"]/-365, color='red', s=400)
st.pyplot(fig)



