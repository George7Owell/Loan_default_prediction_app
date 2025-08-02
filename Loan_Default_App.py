import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import re
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Setting up Home page configuration
st.set_page_config(
    page_title="Loan Default Prediction App",
    layout="wide",
    page_icon="💸",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("loan_Default.csv")
    return df

df = load_data()


# Function to hold pages
def Home_Page():
    st.title("💸 Interactive Loan Default Prediction Web App using Streamlit")
    st.markdown("### Applied Regression and Machine Learning Project")

    # Project Overview Section
    st.markdown("""
    Welcome to our multi-page web app designed to **predict loan default amounts** based on user input and historical data.  
    This project walks through the full machine learning pipeline using **Python**, **Streamlit**, and **Scikit-learn**, from data exploration to prediction.

    This interactive app demonstrates an end-to-end data science workflow using a real-world dataset of loan applicants, including:

    - 📥 Data Import and Overview  
    - 🧹 Data Preprocessing  
    - 🔍 Feature Selection (Best Subset)  
    - 🤖 Model Training using Ridge Regression  
    - 📈 Model Evaluation (Cross-Validation, RMSE, R²)  
    - 🧮 Interactive Prediction Tool  
    - 📊 Results Interpretation and Summary
    ---
    ### 🚀 Instructions:

    1. Use the sidebar menu on the left to navigate between the pages.
    2. Start from **"1. Data Upload and Overview"**.
    3. Follow each step in sequence for best results.

    ---
    ### 📌 Dataset Information:
    - Source: [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
    - Target variable: `Status` (indicating default)


    ### 🎯 **Objective**
    Build an end-to-end machine learning system that:
    - Cleans and preprocesses loan data
    - Selects the most predictive features
    - Trains a **Ridge Regression model**
    - Evaluates its performance
    - Allows users to predict **loan default amounts** interactively
    """)

    ### Defining the metadata
    st.markdown("""
    ### **Data Atributes** """)
    data_dict = [
        {"Column": "ID", "Data Type": "int", "Model Role": "Ignore", "Description": "Unique record ID."},
        {"Column": "year", "Data Type": "int", "Model Role": "Ignore", "Description": "Year of loan application."},
        {"Column": "loan_limit", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Loan amount limit type."},
        {"Column": "Gender", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Gender of primary applicant."},
        {"Column": "approv_in_adv", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Was loan approved in advance?"},
        {"Column": "loan_type", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Loan product type."},
        {"Column": "loan_purpose", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Purpose of the loan."},
        {"Column": "Credit_Worthiness", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Applicant credit profile."},
        {"Column": "open_credit", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Whether applicant has open credit lines."},
        {"Column": "business_or_commercial", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Is the loan for business or commercial use?"},
        {"Column": "loan_amount", "Data Type": "float", "Model Role": "Numerical",
            "Description": "Total amount requested."},
        {"Column": "rate_of_interest", "Data Type": "float", "Model Role": "Numerical",
             "Description": "Interest rate on the loan."},
        {"Column": "Interest_rate_spread", "Data Type": "float", "Model Role": "Numerical",
             "Description": "Difference in interest rate and benchmark."},
        {"Column": "Upfront_charges", "Data Type": "float", "Model Role": "Numerical",
             "Description": "Initial fees paid upfront."},
        {"Column": "term", "Data Type": "int", "Model Role": "Numerical",
             "Description": "Loan repayment period (months)."},
        {"Column": "Neg_ammortization", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Is there negative amortization?"},
        {"Column": "interest_only", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Interest-only loan?"},
        {"Column": "lump_sum_payment", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Any lump-sum payment options?"},
        {"Column": "property_value", "Data Type": "float", "Model Role": "Numerical",
             "Description": "Market value of the property."},
        {"Column": "construction_type", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Construction classification of property."},
        {"Column": "occupancy_type", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Primary, Secondary, or Investment home."},
        {"Column": "Secured_by", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Type of security (e.g., home, land)."},
        {"Column": "total_units", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Total dwelling units."},
        {"Column": "income", "Data Type": "float", "Model Role": "Numerical",
             "Description": "Applicant's monthly income."},
        {"Column": "credit_type", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Main credit reporting agency."},
        {"Column": "Credit_Score", "Data Type": "float", "Model Role": "Numerical",
             "Description": "Numerical credit score."},
        {"Column": "co-applicant_credit_type", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Co-applicant's credit agency."},
        {"Column": "age", "Data Type": "object", "Model Role": "Ordinal",
             "Description": "Applicant age range (e.g., 25-34)."},
        {"Column": "submission_of_application", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Was the application submitted online or in person?"},
        {"Column": "LTV", "Data Type": "float", "Model Role": "Numerical", "Description": "Loan to value ratio."},
        {"Column": "Region", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Region where application was submitted."},
        {"Column": "Security_Type", "Data Type": "object", "Model Role": "Categorical",
             "Description": "Form of security for loan (e.g., direct, indirect)."},
        {"Column": "Status", "Data Type": "int", "Model Role": "Target",
             "Description": "Loan status (1 = defaulted, 0 = paid)."},
        {"Column": "dtir1", "Data Type": "float", "Model Role": "Numerical", "Description": "Debt-to-Income Ratio."}
    ]

    metadata_df = pd.DataFrame(data_dict)

    st.dataframe(metadata_df, use_container_width=True, height=600)

    st.info("This table helps to understand what each column means and how it's used in the prediction model.")

    # Group 5 Members Section
    st.markdown("---")
    st.markdown("### 👥 Project Team")
    team_members = [
    ("Kingsley Sarfo", "22252461", "Project Coordinator, App Design, Preprocessing"),
    ("Francisca Manu Sarpong", "22255796", "Feature Selection, Model Training"),
    ("George Owell", "22256146", "Evaluation, Cross-validation"),
    ("Barima Owiredu Addo", "22254055", "Interactive Prediction UI, Testing"),
    ("Akrobettoe Marcus", "11410687", "Documentation, Deployment")
    ]
