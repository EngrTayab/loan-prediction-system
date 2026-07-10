import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
<style>

.main{
    background:#f4f7fb;
}

h1,h2,h3{
    color:#003366;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 5px 15px rgba(0,0,0,.15);
    margin-bottom:20px;
}

.metric-card{
    background:#0d6efd;
    color:white;
    padding:18px;
    border-radius:15px;
    text-align:center;
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

.success-card{
    background:#d4edda;
    color:#155724;
    padding:20px;
    border-radius:10px;
}

.reject-card{
    background:#f8d7da;
    color:#721c24;
    padding:20px;
    border-radius:10px;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🏦 Loan Prediction")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Loan Prediction",
        "🤖 About Model",
        "📊 Dataset",
        "📈 Feature Guide",
        "👨‍💻 About Developer",
    ],
)

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

st.sidebar.info("""
Algorithm

Gaussian Naive Bayes

Frontend

Streamlit

Backend

Python

Library

Scikit-Learn
""")

# =========================
# HOME PAGE
# =========================

if menu == "🏠 Home":

    st.title("🏦 Loan Prediction System")

    st.write("""
    Welcome to the **Loan Prediction System**.

    This application predicts whether a customer is eligible
    for a loan using Machine Learning.
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
        <div class="metric-card">
        <h3>Model</h3>
        GaussianNB
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
        <div class="metric-card">
        <h3>Features</h3>
        11
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
        <div class="metric-card">
        <h3>Dataset</h3>
        Loan Prediction
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("✨ Project Features")

    st.markdown("""
✅ Loan Approval Prediction

✅ Machine Learning Model

✅ Fast Prediction

✅ User Friendly Interface

✅ Modern Dashboard

✅ Streamlit Web Application

✅ Professional UI
""")

# =========================
# LOAN PREDICTION
# =========================

elif menu == "🔍 Loan Prediction":

    st.title("🔍 Loan Eligibility Prediction")

    st.write("Fill in the applicant information below.")

    left, right = st.columns(2)

    with left:

        gender = st.selectbox(
            "Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x]
        )

        married = st.selectbox(
            "Married", [0, 1], format_func=lambda x: ["No", "Yes"][x]
        )

        dependents = st.selectbox(
            "Dependents", [0, 1, 2, 3], format_func=lambda x: ["0", "1", "2", "3+"][x]
        )

        education = st.selectbox(
            "Education", [0, 1], format_func=lambda x: ["Graduate", "Not Graduate"][x]
        )

        self_employed = st.selectbox(
            "Self Employed", [0, 1], format_func=lambda x: ["No", "Yes"][x]
        )

        property_area = st.selectbox(
            "Property Area",
            [0, 1, 2],
            format_func=lambda x: ["Rural", "Semiurban", "Urban"][x],
        )

    with right:

        applicant_income = st.number_input(
            "Applicant Income", min_value=0, value=5000, step=100
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0, value=0, step=100
        )

        loan_amount = st.number_input("Loan Amount", min_value=1, value=120)

        loan_term = st.number_input(
            "Loan Amount Term", min_value=12, value=360, step=12
        )

        credit_history = st.selectbox(
            "Credit History", [1, 0], format_func=lambda x: "Good" if x == 1 else "Bad"
        )

    st.markdown("---")

    predict = st.button("🚀 Predict Loan Status")

    if predict:

        X = pd.DataFrame(
            {
                "Gender": [gender],
                "Married": [married],
                "Dependents": [dependents],
                "Education": [education],
                "Self_Employed": [self_employed],
                "ApplicantIncome": [applicant_income],
                "CoapplicantIncome": [coapplicant_income],
                "LoanAmount": [loan_amount],
                "Loan_Amount_Term": [loan_term],
                "Credit_History": [credit_history],
                "Property_Area": [property_area],
            }
        )

        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]

        probability = model.predict_proba(X_scaled)[0]

        confidence = max(probability) * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == "Y":

            st.markdown(
                f"""
            <div class="success-card">

            <h2>✅ Loan Approved</h2>

            <h3>Confidence : {confidence:.2f}%</h3>

            </div>
            """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
            <div class="reject-card">

            <h2>❌ Loan Rejected</h2>

            <h3>Confidence : {confidence:.2f}%</h3>

            </div>
            """,
                unsafe_allow_html=True,
            )

        st.progress(confidence / 100)

        st.subheader("Applicant Information")

        st.dataframe(X, use_container_width=True)

        st.session_state.history.append(
            {
                "Time": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Prediction": "Approved" if prediction == "Y" else "Rejected",
                
                "Confidence": round(confidence, 2),
                
                "Gender": "Male" if gender == 1 else "Female",

                "Married": "Yes" if married == 1 else "No",

                "Dependents": ["0", "1", "2", "3+"][dependents],

                "Education": "Graduate" if education == 0 else "Not Graduate",

                "Self Employed": "Yes" if self_employed == 1 else "No",

                "Applicant Income": applicant_income,

                "Coapplicant Income": coapplicant_income,

                "Loan Amount": loan_amount,

                "Loan Amount Term": loan_term,

                "Credit History": "Good" if credit_history == 1 else "Bad",

                "Property Area": ["Rural", "Semiurban", "Urban"][property_area],
                
                
            }
        )

        if st.session_state.history:

            st.subheader("Prediction History")

            history_df = pd.DataFrame(st.session_state.history)

            st.dataframe(history_df, use_container_width=True)

            csv = history_df.to_csv(index=False).encode()

            st.download_button(
                "📥 Download History", csv, "prediction_history.csv", "text/csv"
            )

# =========================
# ABOUT MODEL
# =========================

elif menu == "🤖 About Model":

    st.title("🤖 About Machine Learning Model")

    st.markdown("""
### Gaussian Naive Bayes

Gaussian Naive Bayes is a supervised machine learning classification
algorithm used to predict whether a loan should be approved or rejected.

The algorithm assumes that every feature contributes independently to the prediction.
""")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Algorithm", "GaussianNB")

    with c2:
        st.metric("Learning", "Supervised")

    with c3:
        st.metric("Features", "11")

    st.markdown("---")

    st.subheader("Advantages")

    st.success("""
✔ Fast Prediction

✔ Simple Algorithm

✔ High Performance

✔ Good for Small Datasets

✔ Low Memory Usage
""")

    st.subheader("Model Pipeline")

    st.info("""
Dataset

↓

Data Cleaning

↓

Label Encoding

↓

Standard Scaling

↓

Gaussian Naive Bayes

↓

Prediction
""")


# =========================
# DATASET
# =========================

elif menu == "📊 Dataset":

    st.title("📊 Dataset Information")

    st.write("""
This project uses the Loan Prediction Dataset.

The objective is to predict whether
a customer is eligible for a loan.
""")

    st.metric("Rows", "614")
    st.metric("Features", "11")

    dataset = pd.DataFrame(
        {
            "Feature": [
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self Employed",
                "Applicant Income",
                "Coapplicant Income",
                "Loan Amount",
                "Loan Amount Term",
                "Credit History",
                "Property Area",
            ],
            "Type": [
                "Categorical",
                "Categorical",
                "Categorical",
                "Categorical",
                "Categorical",
                "Numeric",
                "Numeric",
                "Numeric",
                "Numeric",
                "Numeric",
                "Categorical",
            ],
        }
    )

    st.dataframe(dataset, use_container_width=True)


# =========================
# FEATURE GUIDE
# =========================

elif menu == "📈 Feature Guide":

    st.title("📈 Feature Guide")

    features = {
        "Gender": "Applicant Gender",
        "Married": "Marital Status",
        "Dependents": "Number of Dependents",
        "Education": "Education Level",
        "Self Employed": "Employment Status",
        "Applicant Income": "Monthly Income",
        "Coapplicant Income": "Co Applicant Income",
        "Loan Amount": "Requested Loan Amount",
        "Loan Amount Term": "Loan Duration",
        "Credit History": "Previous Loan History",
        "Property Area": "Property Location",
    }

    for key, value in features.items():

        with st.expander(key):

            st.write(value)


# =========================
# ABOUT DEVELOPER
# =========================

elif menu == "👨‍💻 About Developer":

    st.title("👨‍💻 About Developer")

    st.markdown("""
# Muhammad Tayyab Asif

### BS Software Engineering

Islamia University Bahawalpur

Machine Learning & Python Developer
""")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Language", "Python")

    with c2:
        st.metric("Framework", "Streamlit")

    with c3:
        st.metric("Model", "GaussianNB")

    st.markdown("---")

    st.subheader("Skills")

    st.success("""
✔ Python

✔ Machine Learning

✔ Streamlit

✔ Pandas

✔ NumPy

✔ Scikit-Learn

✔ HTML

✔ CSS

✔ Bootstrap

✔ JavaScript
""")

    st.subheader("Project")

    st.info("""
Loan Prediction System

Algorithm:
Gaussian Naive Bayes

Frontend:
Streamlit

Backend:
Python

Libraries:
Pandas
NumPy
Scikit-Learn
Joblib
""")

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;color:gray'>

© 2026 Loan Prediction System

Instructor <b>Sir Zafar Iqbal</b>(AI-ML Trainer)

Developed by <b>Muhammad Tayyab Asif</b>

BS Software Engineering(6th Semester)

Islamia University Bahawalpur

Powered by Streamlit & Scikit-Learn

</div>
""",
    unsafe_allow_html=True,
)
