import streamlit as st
import pickle
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Benign result */
    .benign {
        background-color: #d9f2df;
        border: 2px solid #28a745;
        color: #145c25 !important;
    }

    .benign h2,
    .benign p {
        color: #145c25 !important;
    }

    /* Malignant result */
    .malignant {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24 !important;
    }

    .malignant h2,
    .malignant p {
        color: #721c24 !important;
    }

    .probability {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎗️ Breast Cancer Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Logistic Regression model using <b>radius_mean</b> as the only input feature.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("ℹ️ About this model"):

    st.write(
        """
        This application demonstrates a Logistic Regression model
        trained to classify breast tumor samples as **Benign** or
        **Malignant**.

        The model uses only one feature:

        **radius_mean**
        """
    )

    st.write("### Model details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Logistic Regression")

    with col2:
        st.metric("Features Used", "1")

    with col3:
        st.metric("Feature", "radius_mean")


# =========================================================
# INPUT SECTION
# =========================================================

st.header("🔢 Enter Tumor Measurement")

st.write(
    "Enter the `radius_mean` value obtained from the sample."
)

# Changed back to 0–50
radius_mean = st.number_input(
    "Radius Mean",
    min_value=0.0,
    max_value=50.0,
    value=14.0,
    step=0.1,
    format="%.2f",
    help="Enter the radius_mean value."
)


# =========================================================
# INPUT SUMMARY
# =========================================================

st.markdown("### 📋 Input Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Radius Mean",
        f"{radius_mean:.2f}"
    )

with col2:
    st.metric(
        "Feature Count",
        "1"
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[radius_mean]],
        columns=["radius_mean"]
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    class_index = list(model.classes_).index(1)

    malignant_probability = probabilities[class_index]

    # =====================================================
    # RESULT
    # =====================================================

    st.header("📊 Prediction Result")

    if prediction == 1:

        st.markdown(
            """
            <div class="result-box malignant">
                <h2>⚠️ Malignant</h2>
                <p>The model classified this sample as malignant.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-box benign">
                <h2>✅ Benign</h2>
                <p>The model classified this sample as benign.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # PROBABILITY
    # =====================================================

    st.subheader("Malignant Probability")

    st.markdown(
        f"""
        <div class="probability">
            {malignant_probability * 100:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(malignant_probability))

    # =====================================================
    # PROBABILITY BREAKDOWN
    # =====================================================

    st.subheader("Probability Breakdown")

    benign_probability = 1 - malignant_probability

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Benign",
            f"{benign_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Malignant",
            f"{malignant_probability * 100:.2f}%"
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Test Accuracy",
        "84.21%"
    )

with col2:
    st.metric(
        "ROC-AUC",
        "93.65%"
    )

st.caption(
    "Performance values are based on the 80/20 train-test split "
    "used during model development."
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.warning(
    """
    **Important:** This application is an educational machine-learning
    demonstration. It uses only `radius_mean`, which is not sufficient
    by itself for clinical diagnosis. The prediction should not be used
    as medical advice or as a substitute for evaluation by a qualified
    healthcare professional.
    """
)

st.caption(
    "Built with Python • Scikit-learn • Streamlit"
)
