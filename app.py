import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

model = tf.keras.models.load_model(
    "titanic_ann_model.keras"
)

scaler = joblib.load(
    "scaler.pkl"
)

# ==========================================
# HEADER SECTION
# ==========================================

st.markdown(
"""
# 🚢 Titanic Survival Prediction System

### Deep Learning Based Passenger Survival Prediction
"""
)

st.markdown("---")

# ==========================================
# PROJECT DESCRIPTION
# ==========================================

with st.container():

    st.info(
    """
This application predicts passenger survival probability
using an Artificial Neural Network (ANN)
built using TensorFlow.

### Features Used
• Passenger Class

• Age

• Fare

### Model Workflow

Input Data → Preprocessing → ANN Model → Prediction
"""
)

st.markdown("---")

# ==========================================
# INPUT AREA
# ==========================================

st.subheader("Passenger Information")

col1,col2,col3 = st.columns(3)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

with col2:

    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=120.0
    )

st.markdown("---")

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Survival"):

    user_input = np.array(
        [[
            pclass,
            age,
            fare
        ]]
    )

    # SAME PREPROCESSING

    processed = scaler.transform(
        user_input
    )

    prediction = model.predict(
        processed,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    non_survival = 1-probability

    confidence = max(
        probability,
        non_survival
    )

    # PREDICTION LOGIC

    if probability > 0.5:

        result = "Survived ✅"

    else:

        result = "Not Survived ❌"

    st.subheader("Prediction Results")

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Prediction",
        result
    )

    c2.metric(
        "Survival Probability",
        f"{probability:.2%}"
    )

    c3.metric(
        "Confidence",
        f"{confidence:.2%}"
    )

    st.markdown("---")

    # =====================================
    # VISUALIZATION
    # =====================================

    st.subheader("Prediction Visualization")

    fig,ax = plt.subplots(
        figsize=(5,5)
    )

    labels = [
        "Survival",
        "Non Survival"
    ]

    values = [
        probability,
        non_survival
    ]

    colors = [
        "green",
        "red"
    ]

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors
    )

    st.pyplot(fig)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.success(
"TensorFlow ANN Deployment Successful"
)