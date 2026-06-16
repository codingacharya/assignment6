import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Loan Eligibility Dashboard", layout="wide")

# 1. GENERATE SAMPLE DATASET
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'Loan_ID': [f'LP{i:05d}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.6, 0.15, 0.15, 0.10]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.8, 0.2]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'ApplicantIncome': np.random.randint(1500, 10000, n_samples),
        'CoapplicantIncome': np.random.choice([0.0, 1500.0, 2500.0, 4000.0], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'LoanAmount': np.random.randint(50, 300, n_samples),
        'Loan_Amount_Term': np.random.choice([120, 180, 240, 360], n_samples, p=[0.05, 0.05, 0.1, 0.8]),
        'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.85, 0.15]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.3, 0.4, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Synthesize realistic logic for Loan_Status (Approval)
    # Credit history = 1 and decent income increases approval drastically
    total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Loan_Status'] = np.where(
        (df['Credit_History'] == 1.0) & (total_income / df['LoanAmount'] > 25), 
        'Y', 
        np.random.choice(['Y', 'N'], n_samples, p=[0.2, 0.8])
    )
    return df

df_raw = load_sample_data()

# 2. SIDEBAR - MAXIMUM FILTERS
st.sidebar.header("📊 Advanced Dataset Filters")

# Categorical Multiselect Filters (Initialized with all unique values as default)
gender_f = st.sidebar.multiselect("Gender", df_raw['Gender'].unique(), default=df_raw['Gender'].unique())
married_f = st.sidebar.multiselect("Married", df_raw['Married'].unique(), default=df_raw['Married'].unique())
dependents_f = st.sidebar.multiselect("Dependents", df_raw['Dependents'].unique(), default=df_raw['Dependents'].unique())
education_f = st.sidebar.multiselect("Education", df_raw['Education'].unique(), default=df_raw['Education'].unique())
employment_f = st.sidebar.multiselect("Self Employed", df_raw['Self_Employed'].unique(), default=df_raw['Self_Employed'].unique())
property_f = st.sidebar.multiselect("Property Area", df_raw['Property_Area'].unique(), default=df_raw['Property_Area'].unique())
status_f = st.sidebar.multiselect("Loan Status (Approved?)", df_raw['Loan_Status'].unique(), default=df_raw['Loan_Status'].unique())

# Numerical Range Slider Filters
min_income, max_income = int(df_raw['ApplicantIncome'].min()), int(df_raw['ApplicantIncome'].max())
income_f = st.sidebar.slider("Applicant Income Range", min_income, max_income, (min_income, max_income))

min_loan, max_loan = int(df_raw['LoanAmount'].min()), int(df_raw['LoanAmount'].max())
loan_f = st.sidebar.slider("Loan Amount Range ($K)", min_loan, max_loan, (min_loan, max_loan))

# Radio button for Credit History
credit_f = st.sidebar.radio("Credit History Requirements", ["All", "Clear History (1.0)", "No History/Bad (0.0)"])

# 3. APPLY FILTERS TO DATASET
df_filtered = df_raw[
    (df_raw['Gender'].isin(gender_f)) &
    (df_raw['Married'].isin(married_f)) &
    (df_raw['Dependents'].isin(dependents_f)) &
    (df_raw['Education'].isin(education_f)) &
    (df_raw['Self_Employed'].isin(employment_f)) &
    (df_raw['Property_Area'].isin(property_f)) &
    (df_raw['Loan_Status'].isin(status_f)) &
    (df_raw['ApplicantIncome'].between(income_f[0], income_f[1])) &
    (df_raw['LoanAmount'].between(loan_f[0], loan_f[1]))
]

if credit_f == "Clear History (1.0)":
    df_filtered = df_filtered[df_filtered['Credit_History'] == 1.0]
elif credit_f == "No History/Bad (0.0)":
    df_filtered = df_filtered[df_filtered['Credit_History'] == 0.0]


# 4. MAIN DASHBOARD LAYOUT
st.title("🏦 Automated Loan Eligibility Predictor & Explorer")
st.markdown("This dashboard showcases the generated sample dataset, filtered metrics, and an active predictive simulator machine.")

# Top metrics bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Applications (Filtered)", len(df_filtered))
col2.metric("Approval Rate", f"{round((df_filtered['Loan_Status'] == 'Y').sum() / max(len(df_filtered), 1) * 100, 1)}%")
col3.metric("Avg Applicant Income", f"${int(df_filtered['ApplicantIncome'].mean() if len(df_filtered) > 0 else 0):,}")
col4.metric("Avg Requested Loan", f"${int(df_filtered['LoanAmount'].mean() if len(df_filtered) > 0 else 0)*1000:,}")

st.write("---")

# Main split layout: Data Explorer vs Live Predictor Simulator
tab1, tab2 = st.tabs(["🔍 Dataset Explorer", "🧠 Live Predictive Simulator"])

with tab1:
    st.subheader("Filtered Application Data")
    st.dataframe(df_filtered, use_container_width=True)
    st.caption(f"Showing {len(df_filtered)} out of {len(df_raw)} records based on sidebar parameters.")

with tab2:
    st.subheader("Test Real-Time Predictive AI Decisioning")
    st.write("Modify the criteria below to see if the Machine Learning engine would approve or reject the loan application.")
    
    # Train a fast mockup model backend on the data
    @st.cache_resource
    def train_backend_model(data):
        df_model = data.copy().dropna()
        le_dict = {}
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
            
        X = df_model.drop(columns=['Loan_ID', 'Loan_Status'])
        y = df_model['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)
        
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        return clf, le_dict, X.columns.tolist()

    clf, le_dict, feature_names = train_backend_model(df_raw)

    # Simulator Inputs Form
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        s_gender = st.selectbox("Sim: Gender", ["Male", "Female"])
        s_married = st.selectbox("Sim: Married", ["Yes", "No"])
        s_dependents = st.selectbox("Sim: Dependents", ["0", "1", "2", "3+"])
    with sim_col2:
        s_edu = st.selectbox("Sim: Education", ["Graduate", "Not Graduate"])
        s_emp = st.selectbox("Sim: Self Employed", ["Yes", "No"])
        s_prop = st.selectbox("Sim: Property Area", ["Urban", "Semiurban", "Rural"])
    with sim_col3:
        s_income = st.number_input("Sim: Applicant Income ($)", value=5000)
        s_co_income = st.number_input("Sim: Coapplicant Income ($)", value=1500)
        s_loan = st.number_input("Sim: Loan Amount (In Thousands $)", value=150)
        s_term = st.selectbox("Sim: Term (Days)", [360, 240, 180, 120])
        s_credit = st.selectbox("Sim: Credit History Satisfied?", [1.0, 0.0])

    # Prediction Logic execution
    input_data = pd.DataFrame([{
        'Gender': s_gender, 'Married': s_married, 'Dependents': s_dependents,
        'Education': s_edu, 'Self_Employed': s_emp, 'ApplicantIncome': s_income,
        'CoapplicantIncome': float(s_co_income), 'LoanAmount': s_loan,
        'Loan_Amount_Term': s_term, 'Credit_History': s_credit, 'Property_Area': s_prop
    }])

    # Encode input properties using saved LabelEncoders
    for col in le_dict:
        try:
            input_data[col] = le_dict[col].transform(input_data[col].astype(str))
        except ValueError:
            input_data[col] = 0 # Fallback safety handling

    # Ensure feature sequence matching 
    input_data = input_data[feature_names]
    
    prediction = clf.predict(input_data)[0]
    prob = clf.predict_proba(input_data)[0][1]

    st.write("---")
    if prediction == 1:
        st.success(f"🎉 **LOAN APPROVED!** (Model Confidence: {prob*100:.1f}%)")
    else:
        st.error(f"❌ **LOAN REJECTED.** (Model Confidence: {(1-prob)*100:.1f}%)")