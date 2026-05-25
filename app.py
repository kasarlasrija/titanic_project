import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Titanic AI Prediction",
    page_icon="🚢",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""

<style>

.main{
background-color:#0E1117;
}

.title{
font-size:42px;
font-weight:bold;
color:#00D4FF;
}

.subtitle{
font-size:20px;
color:#D3D3D3;
}

.metric-card{

padding:15px;

border-radius:10px;

background:#1E1E1E;

}

</style>

""",unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource

def load_model():

    model=tf.keras.models.load_model(
        "titanic_ann_model.keras"
    )

    scaler=joblib.load(
        "scaler.pkl"
    )

    return model,scaler

try:

    model,scaler=load_model()

except Exception as e:

    st.error(
    f"Model Loading Error : {e}"
    )

    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
"Navigation"
)

menu=st.sidebar.radio(

"Menu",

[
"🏠 Home",

"📖 Project Info",

"🚢 Prediction",

"🧠 Model Details"

]

)

# =====================================================
# HOME
# =====================================================

if menu=="🏠 Home":

    st.markdown(

    """

<div class='title'>

🚢 Titanic Survival Prediction System

</div>

""",

unsafe_allow_html=True

)

    st.markdown(

"""
<div class='subtitle'>

Deep Learning Based Passenger Survival Prediction

</div>
""",

unsafe_allow_html=True

)

    st.divider()

    c1,c2,c3=st.columns(3)

    c1.metric(
    "Framework",
    "TensorFlow"
    )

    c2.metric(
    "Model",
    "ANN"
    )

    c3.metric(
    "Features",
    "3"
    )

    st.success(

"""
AI powered prediction system using

✔ TensorFlow

✔ ANN

✔ Streamlit

✔ Deep Learning
"""
)

# =====================================================
# PROJECT INFO
# =====================================================

elif menu=="📖 Project Info":

    st.header(
    "Project Description"
    )

    st.info(

"""
Purpose:

Predict passenger survival.

Features:

• Passenger Class

• Age

• Fare

Workflow:

Input

↓

Preprocessing

↓

ANN Model

↓

Prediction

↓

Visualization
"""
)

# =====================================================
# MODEL DETAILS
# =====================================================

elif menu=="🧠 Model Details":

    st.header(
    "ANN Architecture"
    )

    st.code(

"""
Input Layer : 3

Hidden Layer 1 : Dense(16)

Activation : ReLU

Hidden Layer 2 : Dense(8)

Activation : ReLU

Output Layer : Dense(1)

Activation : Sigmoid
"""
)

# =====================================================
# PREDICTION
# =====================================================

elif menu=="🚢 Prediction":

    st.header(
    "Passenger Details"
    )

    c1,c2,c3=st.columns(3)

    with c1:

        pclass=st.selectbox(

        "Passenger Class",

        [1,2,3]

        )

    with c2:

        age=st.slider(

        "Age",

        1,

        80,

        25

        )

    with c3:

        fare=st.number_input(

        "Fare",

        0.0,

        600.0,

        100.0

        )

    st.divider()

    if st.button(

    "Predict Survival",

    use_container_width=True

    ):

        data=np.array(

        [[

        pclass,

        age,

        fare

        ]]

        )

        processed=scaler.transform(
        data
        )

        prediction=model.predict(

        processed,

        verbose=0

        )

        probability=float(
        prediction[0][0]
        )

        non_survival=1-probability

        confidence=max(

        probability,

        non_survival

        )

        if probability>0.5:

            result="Survived ✅"

        else:

            result="Not Survived ❌"

        st.subheader(
        "Prediction Output"
        )

        a,b,c=st.columns(3)

        a.metric(

        "Result",

        result

        )

        b.metric(

        "Survival Probability",

        f"{probability:.2%}"

        )

        c.metric(

        "Confidence",

        f"{confidence:.2%}"

        )

        st.write(
        "Probability Meter"
        )

        st.progress(
        probability
        )

        fig,ax=plt.subplots(
        figsize=(5,5)
        )

        ax.pie(

        [

        probability,

        non_survival

        ],

        labels=[

        "Survival",

        "Non Survival"

        ],

        colors=[

        "green",

        "red"

        ],

        autopct="%1.1f%%"

        )

        st.pyplot(fig)

# =====================================================
# FOOTER
# =====================================================

st.sidebar.markdown("---")

st.sidebar.success(
"ANN Deployment Successful"
)
