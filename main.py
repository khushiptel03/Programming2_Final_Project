import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

# Load the model
def load_model():
    return joblib.load('Programming2_Final_Project')

model = load_model()

# Title and header
st.title("Welcome to Khushi’s Application")
st.header("LinkedIn Usage Prediction App")

# Description
st.write("""
Based on your social media habits, we will predict if you are a LinkedIn user or not!
Let's get started!
""")

income_level = st.selectbox(
    "What is your household income level?",
    ["Less than $10,000", "$10,000 to $49,999", "$50,000 to $99,999", "$100,000 or more"]
)

education_level = st.selectbox(
    "What is your highest level of school/degree completed?",
    ["Less than high school", "High school", "Some college", "Associate degree", "Bachelor's degree", "Postgraduate degree"]
)

parent_status = st.radio("Are you a parent of a child under 18?", ["Yes", "No", "Don't know", "Refused"])

marital_status = st.selectbox(
    "What is your marital status?",
    ["Married", "Widowed", "Divorced", "Separated", "Never married", "Living with a partner"]
)

gender = st.radio("What is your gender?", ["Male", "Female", "Other", "Don't know", "Refused"])

age = st.slider("What is your age?", min_value=18, max_value=97, step=1)

# Helper function to preprocess user input into a format the model can understand
def preprocess_input(income_level, education_level, parent_status, marital_status, gender, age):
    # Convert categorical data to numeric values (you may need to adjust mappings)
    income_mapping = {
        "Less than $10,000": 1, "$10,000 to $49,999": 2, "$50,000 to $99,999": 3, "$100,000 or more": 4
    }
    education_mapping = {
        "Less than high school": 1, "High school": 2, "Some college": 3, "Associate degree": 4,
        "Bachelor's degree": 5, "Postgraduate degree": 6
    }
    marital_mapping = {
        "Married": 1, "Widowed": 2, "Divorced": 3, "Separated": 4, "Never married": 5, "Living with a partner": 6
    }
    gender_mapping = {
        "Male": 1, "Female": 0, "Other": 2
    }

    # Preprocess the inputs to numeric format
    income = income_mapping.get(income_level, 0)
    education = education_mapping.get(education_level, 0)
    marital = marital_mapping.get(marital_status, 0)
    female = gender_mapping.get(gender, 0)

    parent = 1 if parent_status == "Yes" else 0

    # Return as numpy array
    return np.array([[income, education, parent, marital, female, age]])

# Submit button to make the prediction
if st.button("Submit"):
    # Preprocess the user input for prediction
    user_data = preprocess_input(income_level, education_level, parent_status, marital_status, gender, age)

    # Make prediction using the trained model
    prediction = model.predict(user_data)

    # Display the prediction result
    if prediction == 1:
        st.write("🎉 You are likely to be a LinkedIn user!")
    else:
        st.write("❌ You are unlikely to be a LinkedIn user.")

    # Optional: Show a simple visualization based on prediction
    fig = go.Figure(go.Bar(x=["LinkedIn User", "Non-LinkedIn User"], y=[1, 1], name="Prediction"))
    st.plotly_chart(fig)
