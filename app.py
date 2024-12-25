# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Load the saved model and scaler
with open('random_forest_model.pkl', 'rb') as f:
    best_rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load iris dataset for reference
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# Streamlit app title
st.title("Iris Flower Prediction")
st.write("This app predicts the species of Iris flowers based on the input features.")

# Add input fields for the four features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

# Prepare the input data for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

# Predict the species when the button is clicked
if st.button('Predict'):
    prediction = best_rf_model.predict(input_data_scaled)
    predicted_species = iris.target_names[prediction][0]
    st.write(f"Predicted Species: {predicted_species}")

# Optional: Display the model's feature importance
st.write("### Feature Importance (Random Forest)")

# Accessing the feature importances from the RandomForestClassifier
feature_importances = best_rf_model.named_steps['randomforestclassifier'].feature_importances_
features = iris.feature_names

# Plot feature importance
fig, ax = plt.subplots()
ax.barh(features, feature_importances)
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance for Random Forest')
st.pyplot(fig)

# Optional: Display the Iris dataset for reference
st.write("### Iris Dataset Sample")
st.dataframe(iris_df.head())
