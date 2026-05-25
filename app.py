import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Titanic AI Prediction",
    page_icon="🚢",
    layout="wide"
)

# ==================================
# CUSTOM STYLING
# ==================================

st.markdown("""
<style>

.main{
background-color:#0E1117;
}

.title{
font-size:40px;
font-weight:bold;
color:#00D4FF;
}

.subtitle{
font-size:18px;
color:#C0C0C0;
}

.card{
padding:20px;
border-radius:15px;
background-color:#1E1E1E;
box-shadow:2px 2px 8px gray;
}

</style>
""",unsafe_allow_html=True)

# ==================================
# LOAD MODEL
# ==================================

@st.cache_resource
def load_files():

    model=tf.keras.models.load_model(
        "titanic_ann_model.keras"
    )

    scaler=joblib.load(
        "scaler.pkl"
    )

    return model,scaler

try:

    model,scaler=load_files()

except:

    st.error(
"""
Required Files Missing

Place:

• titanic_ann_model.keras

• scaler.pkl

inside project folder
"""
)
    st.stop()

# ==================================
# SIDEBAR MENU
# ==================================

st.sidebar.image(
"https://cdn-icons-png.flaticon.com/512/2784/2784487.png",
width=100
)

st.sidebar.title(
"Navigation"
)

menu=st.sidebar.radio(

"Go To",

[
"🏠 Home",
"📖 Project Info",
"🚢 Prediction",
"🧠 Model Details"
]

)

# ==================================
# HOME
# ==================================

if menu=="🏠 Home":

    st.markdown(
"""
<div class='title'>
🚢 Titanic Survival Prediction
</div>
""",
unsafe_allow_html=True
)

    st.markdown(
"""
<div class='subtitle'>
Deep Learning Based Passenger Survival Prediction System
</div>
""",
unsafe_allow_html=True
)

    st.divider()

    col1,col2=st.columns([2,1])

    with col1:

        st.success(
"""
Predict passenger survival using

✔ Artificial Neural Network

✔ TensorFlow Deep Learning

✔ Streamlit Deployment

✔ AI Powered Prediction
"""
)

    with col2:

        st.metric(
        "Model Type",
        "ANN"
        )

        st.metric(
        "Features",
        "3"
        )

        st.metric(
        "Framework",
        "TensorFlow"
        )

# ==================================
# PROJECT INFO
# ==================================

elif menu=="📖 Project Info":

    st.header(
    "Project Description"
    )

    st.info(
"""
Purpose:

Predict passenger survival probability
during emergency situations.

Features Used:

• Passenger Class

• Age

• Fare

Workflow:

Input Data

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

# ==================================
# MODEL DETAILS
# ==================================

elif menu=="🧠 Model Details":

    st.header(
    "Neural Network Architecture"
    )

    st.write(
"""
Input Layer → 3 Neurons

Hidden Layer → Dense Layer

Activation → ReLU

Output Layer → Sigmoid

Loss Function:

Binary Crossentropy

Optimizer:

Adam

Framework:

TensorFlow / Keras
"""
)

    st.code(
"""
Input(3)

↓

Dense(16,activation='relu')

↓

Dense(8,activation='relu')

↓

Dense(1,activation='sigmoid')
"""
)

# ==================================
# PREDICTION PAGE
# ==================================

elif menu=="🚢 Prediction":

    st.header(
    "Passenger Information"
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

        pred=model.predict(

        processed,

        verbose=0

        )

        prob=float(
        pred[0][0]
        )

        non=1-prob

        confidence=max(
        prob,
        non
        )

        result=(
        "Survived ✅"
        if prob>0.5
        else
        "Not Survived ❌"
        )

        st.success(
        "Prediction Generated"
        )

        a,b,c=st.columns(3)

        a.metric(

        "Prediction",

        result

        )

        b.metric(

        "Probability",

        f"{prob:.2%}"

        )

        c.metric(

        "Confidence",

        f"{confidence:.2%}"

        )

        st.write(
        "Survival Probability"
        )

        st.progress(
        prob
        )

        fig,ax=plt.subplots(
        figsize=(5,5)
        )

        ax.pie(

        [prob,non],

        labels=[

        "Survival",

        "Non Survival"

        ],

        autopct="%1.1f%%",

        colors=[

        "green",

        "red"

        ]

        )

        st.pyplot(fig)

# ==================================
# FOOTER
# ==================================

st.sidebar.markdown("---")

st.sidebar.caption(
"AI Deep Learning Project"
)
