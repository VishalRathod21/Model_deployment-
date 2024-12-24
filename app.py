import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the saved model and scaler
model = joblib.load('iris_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

# Load dataset for reference
iris = load_iris()

# Streamlit app
st.title("Iris Flower Prediction")
st.write("Predict the type of Iris flower based on its features. Enter the measurements below and click Predict.")

# Use columns for input fields to improve layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, value=5.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, value=1.4)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, value=3.5)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, value=0.2)

# Prediction button
if st.button("Predict"):
    # Prepare input features
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(scaled_features)[0]
    prediction_proba = model.predict_proba(scaled_features)[0]

    # Display predicted class
    st.write(f"### Predicted Class: **{iris.target_names[prediction]}**")

    # Display prediction probabilities
    st.write("### Prediction Probabilities:")
    proba_df = pd.DataFrame([prediction_proba], columns=iris.target_names)
    st.dataframe(proba_df)

    # Plot prediction probabilities
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=iris.target_names, y=prediction_proba, ax=ax, palette="Set2")
    ax.set_title("Prediction Probabilities")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Iris Classes")
    st.pyplot(fig)

    # Add more help text
    st.write("### How it works:")
    st.write("The model uses the following features to predict the species of the Iris flower:")
    st.write("- **Sepal Length (cm)**")
    st.write("- **Sepal Width (cm)**")
    st.write("- **Petal Length (cm)**")
    st.write("- **Petal Width (cm)**")
    st.write(
        "Once you enter the values, the model provides the predicted Iris flower species along with the probabilities for each class.")

# Footer
st.write("Model Trained using Scikit-learn with Logistic Regression")
