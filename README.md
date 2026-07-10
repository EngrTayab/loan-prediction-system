# 🏦 Loan Prediction System

A modern Machine Learning web application built with **Python**, **Scikit-learn**, and **Streamlit** that predicts whether a loan application is likely to be **Approved** or **Rejected** based on an applicant's financial and personal information.

---

## 📌 Overview

The Loan Prediction System is designed to assist in evaluating loan eligibility using a trained Machine Learning model. The application provides an intuitive interface where users can enter applicant details and instantly receive a prediction along with the model's confidence score.

The project demonstrates the complete Machine Learning workflow, including data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive web application.

---

## ✨ Features

* Modern and responsive Streamlit interface
* Loan approval prediction
* Gaussian Naive Bayes Machine Learning model
* Confidence score for every prediction
* About Model page
* Dataset Information page
* Feature Guide
* About Developer page
* Prediction History with applicant details
* Timestamp for every prediction
* Download prediction history as CSV

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Label Encoding
5. Feature Scaling using StandardScaler
6. Model Training
7. Model Evaluation
8. Model Deployment with Streamlit

---

## 📊 Input Features

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

---

## 🤖 Model Used

* Gaussian Naive Bayes

### Preprocessing

* Label Encoding
* StandardScaler

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## 📂 Project Structure

```text
LoanPrediction/
│
├── app.py
├── best_model.pkl
├── scaler.pkl
├── encoders.pkl
├── requirements.txt
├── README.md
└── dataset.csv
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/LoanPrediction.git
```

Move to the project folder:

```bash
cd LoanPrediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Application Pages

* Home
* Loan Prediction
* About Model
* Dataset Information
* Feature Guide
* About Developer

---

## 📈 Prediction History

The application stores each prediction during the current session, including:

* Applicant Information
* Prediction Result
* Confidence Score
* Date & Time

Users can also download the prediction history as a CSV file.

---

## 🎯 Future Improvements

* User authentication
* Database integration
* Additional Machine Learning models
* Model comparison dashboard
* Cloud deployment
* Real-time analytics

---

## 👨‍💻 Developer

**Muhammad Tayyab Asif**

BS Software Engineering

Islamia University Bahawalpur

---

⭐ If you found this project useful, consider giving it a star on GitHub.

