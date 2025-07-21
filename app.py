import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

# Sidebar inputs
st.sidebar.header("Enter Iris Flower Measurements")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()


st.title(" Iris Flower Species Predictor")
st.write("This app predicts the **species of Iris flowers** using a trained Random Forest model.")

st.subheader("User Input Features")
st.write(input_df)


X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier()
model.fit(X, y)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write(f"**Predicted Species:** {target_names[prediction[0]]}")

st.subheader("Prediction Probabilities")
st.write(pd.DataFrame(prediction_proba, columns=target_names))


st.subheader("Data Distribution")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue=df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'}), palette='deep')
st.pyplot(fig)
