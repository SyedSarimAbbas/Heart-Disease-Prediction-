import streamlit as st
import pandas as pd
import base64
import pickle
import numpy as np
import plotly.graph_objects as go


with open("Heart_disease_prediction_model.pkl","rb") as file:
    model=pickle.load(file)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("image.png")

st.markdown(
    """
    <div style='
        position: absolute;
        top: 50%;
        left: 5%;
        transform: translateY(-50%);
    '>
    </div>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<h1 style='text-align: center; color: red;'> Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.write("Enter your health parameters below to check your risk level.")

# Input form
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp_input = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp_input)

trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol Level (mg/dl)", 100, 400, 200)

fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs_input == "Yes" else 0

restecg_input = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
restecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg_input)

thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)

exang_input = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang_input == "Yes" else 0

oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)

slope_input = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
slope = ["Upsloping", "Flat", "Downsloping"].index(slope_input)

ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)

thal_input = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal_input) + 1 
# Predict button
# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction] * 100

    # Show styled result
    if prediction == 1: 
        st.markdown(
            f"<h4 style='background-color:red; color:white; padding:10px; border-radius:5px;'>⚠ High Risk! Probability: {prob:.2f}%</h4>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h4 style='background-color:green; color:white; padding:10px; border-radius:5px;'>Low Risk! Probability: {prob:.2f}%</h4>", 
            unsafe_allow_html=True
        )

    # Wave Chart (Area Chart)
    x_values = list(range(0, 101))  # 0 to 100%
    y_values = [np.sin((i / 100) * np.pi) * prob / 100 * 100 for i in x_values]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        fill='tozeroy',
        mode='lines',
        line=dict(color="red" if prediction == 1 else "green", width=3),
        name="Risk Wave"
    ))

    # Vertical line for actual probability
    fig.add_vline(
        x=prob,
        line_width=2,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"{prob:.2f}%",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Risk Probability Wave Chart",
        xaxis_title="Probability (%)",
        yaxis_title="Wave Amplitude",
        yaxis=dict(showticklabels=False)
    )

    st.plotly_chart(fig)
