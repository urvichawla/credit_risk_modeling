# Credit Risk Modeling

This project uses machine learning to predict the credit risk of loan applicants. The model categorizes borrowers into four different risk levels (P1-P4) to help banks make data-driven decisions when lending loans.

## Project Overview

The Credit Risk Modeling project uses two datasets:

1. **Internal Loan Dataset**: Contains features related to loan history, including the number of loan accounts, loan type, and activity over the past 6 months.
2. **CIBIL Credit Data**: Includes features like payment history, credit score, and delinquency status of borrowers.

We built a multi-class classification model to predict the credit risk level of borrowers. The model categorizes them into the following risk groups:
- **P1**: Low-risk (Good borrowers)
- **P2**: Medium-low risk
- **P3**: Medium-high risk
- **P4**: High-risk (High chances of default)

## Technologies Used
- **Python**: Programming language for implementing the machine learning model.
- **Streamlit**: Used for creating the interactive web app to display predictions.
- **Scikit-learn**: Machine learning library used to build and evaluate the classification model.
- **NumPy**: For numerical operations and data manipulation.
- **Pandas**: Data processing and cleaning.
- **Matplotlib**: Data visualization.

## How to Run the Project Locally

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Git (for version control)
- Virtual environment (optional, but recommended)

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/urvichawla/credit_risk_modeling.git
2.Navigate into the project directory:

cd credit-risk-modeling

3. Install the requirements:
pip install -r requirements.txt

4. Run the streamlit app:
   streamlit run app.py

