import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st. write("""
# Sample Wine Color Prediction App 
""")
st.sidebar.header("User Input Parameters")

def user_input_features():

    fixed_acidity =st.sidebar.slider('Fixed Acidity', 3.8, 15.9, 5.4)
    volatile_acidity = st.sidebar.slider('Volatile Acidity',0.08, 1.58, 1.4)
    citric_acid = st.sidebar.slider('Citric Acid',0.0, 1.66, 1.3)
    residual_sugar = st.sidebar.slider('Residual Sugar',0.6, 65.8, 1.2)
    chlorides = st.sidebar.slider('Chlorides',0.009, 0.611, 0.4)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide',1.0, 289.0, 38.0)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulphur Dioxide', 6.0, 440.0, 10.0)
    density = st.sidebar.slider('Density', 0.99, 1.04, 1.0)
    pH = st.sidebar.slider('pH', 2.72, 4.01, 3.4)
    sulphates = st.sidebar.slider('Sulphates', 0.22, 2.0, 1.3)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 14.9, 5.0)
    quality = st.sidebar.slider('Quality', 3.0, 9.0, 5.0)

    data = {'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol,
            'quality': quality
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

st.subheader("User Input Parameters")
st.write(df)

data = pd.read_csv("Wine_Quality_Data.csv")

data['color'] = data['color'].map({'red': 1, 'white': 0})

x=data.drop('color',axis=1)
y=data.color

scaler=StandardScaler()
scaler.fit(x)
standardised_df=scaler.transform(x)
df_standard=pd.DataFrame(standardised_df,columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'])
X=df_standard


logreg = LogisticRegression()
logreg.fit(X,y)

##prediction
prediction=logreg.predict(df)
prediction_proba=logreg.predict_proba(df)

if prediction==1:
    prediction="white"
if prediction==0:
    prediction="red"

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)