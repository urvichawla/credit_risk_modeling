import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import time
from scipy.stats import f_oneway
import pickle
import xlsxwriter
import seaborn as sns
import numpy as np

st.sidebar.title("CRisk: Credit Risk Prediction using ML")
st.sidebar.header("📌 Navigation")

page = st.sidebar.selectbox("Select a page", ["Credit Risk Analysis", "EPS", "About"])

if page == "Credit Risk Analysis":
    st.title("Credit Risk Analysis and Prediction")

    start_time = time.time()

    uploaded_file1 = st.file_uploader("Upload Input Dataset 1", type="xlsx")
    uploaded_file2 = st.file_uploader("Upload Input Dataset 2", type="xlsx")
    uploaded_unseen = st.file_uploader("Upload an Unseen Dataset", type="xlsx")

    show_graphs = st.sidebar.checkbox("Show Graphs", value=True)

    if uploaded_file1 and uploaded_file2 and uploaded_unseen:
        with st.spinner("Program is running..."):
            df1 = pd.read_excel(uploaded_file1)
            df2 = pd.read_excel(uploaded_file2)
            
            df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

            columns_to_be_removed = [col for col in df2.columns if df2[df2[col] == -99999].shape[0] > 10000]

            df2 = df2.drop(columns_to_be_removed, axis=1)

            for col in df2.columns:
                df2 = df2[df2[col] != -99999]

            df = pd.merge(df1, df2, how='inner', on='PROSPECTID')
            
            # Chi-Square
            st.subheader("Chi-Square Tests")

            target_column = 'Approved_Flag'

            categorical_columns = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

            chi2_results = []

            if target_column in df.columns:
                for col in categorical_columns:
                    if col in df.columns:
                        chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df[target_column]))
                        chi2_results.append((col, pval))

            chi2_df = pd.DataFrame(chi2_results, columns=["Feature", "p-value"])
            chi2_df["p-value"] = chi2_df["p-value"].apply(lambda x: f"{x:.15e}")

            st.write("Chi-Square Results")
            st.dataframe(chi2_df)

            # Variance Inflation Factor
            vif_results = []

            numeric_columns = []

            for i in df.columns:
                if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
                    numeric_columns.append(i)

            vif_data = df[numeric_columns]

            total_columns = vif_data.shape[1]

            columns_to_be_kept = []

            column_index = 0

            for i in range(total_columns):
                vif_value = variance_inflation_factor(vif_data.values, column_index)
                # st.write(f"{numeric_columns[i]} - VIF: {vif_value}")
                vif_results.append((numeric_columns[i], vif_value))
                if vif_value <= 6:
                    columns_to_be_kept.append(numeric_columns[i])
                    column_index += 1
                else:
                    vif_data = vif_data.drop([numeric_columns[i]], axis=1)

            st.write("VIF Results")
            st.write(pd.DataFrame(vif_results, columns=['Feature', 'VIF']).sort_values(by='VIF'))

            # ANOVA-F
            columns_to_be_kept_numerical = []

            anova_results = []

            for col in columns_to_be_kept:
                a = list(df[col])
                b = list(df['Approved_Flag'])
                
                group_P1 = [value for value, group in zip(a, b) if group == 'P1']
                group_P2 = [value for value, group in zip(a, b) if group == 'P2']
                group_P3 = [value for value, group in zip(a, b) if group == 'P3']
                group_P4 = [value for value, group in zip(a, b) if group == 'P4']

                f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
                # st.write(f"ANOVA for {col} - p-value: {p_value}")
                
                if p_value <= 0.05:
                    columns_to_be_kept_numerical.append(col)

            features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

            df = df[features + ['Approved_Flag']]

            # ['MARITALSTATUS', 'EDUCATION', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']
            df['MARITALSTATUS'].unique()    
            df['EDUCATION'].unique()
            df['GENDER'].unique()
            df['last_prod_enq2'].unique()
            df['first_prod_enq2'].unique()

            df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
            df.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
            df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
            df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
            df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
            df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
            df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3

            df['EDUCATION'].value_counts()
            df['EDUCATION'] = df['EDUCATION'].astype(int)

            df.info()

            df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])

            # stupid bug here [dont need enable_categorical]
            xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  
                                               num_class=4, 
                                               colsample_bytree = 0.9,
                                                learning_rate    = 1.0,
                                                max_depth        = 3,
                                                alpha            = 10.0,
                                                n_estimators     = 100)

            y = df_encoded['Approved_Flag']
            x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

            xgb_classifier.fit(x_train, y_train)
            y_pred = xgb_classifier.predict(x_test)

            if show_graphs:
                importances = xgb_classifier.feature_importances_
                feature_names = x.columns
                indices = np.argsort(importances)[::-1]

                plt.figure(figsize=(10, 6))
                plt.title("Feature Importances")
                plt.bar(range(len(importances)), importances[indices], align="center", color=plt.cm.magma(importances[indices] / max(importances)))
                plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
                plt.xlim([-1, len(importances)])
                st.pyplot(plt)

                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix")
                st.write(cm)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=['p1', 'p2', 'p3', 'p4'], yticklabels=['p1', 'p2', 'p3', 'p4'])
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(plt)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f'Accuracy: {accuracy:.2f}')
            
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

            results_df = pd.DataFrame({
                "Class": ['p1', 'p2', 'p3', 'p4'],
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            })

            st.write(results_df)

            # another bug here [fixed from refactoring]
            df_unseen = pd.read_excel(uploaded_unseen)

            education_map = {
                'SSC': 1,
                '12TH': 2,
                'GRADUATE': 3,
                'UNDER GRADUATE': 3,
                'POST-GRADUATE': 4,
                'OTHERS': 1,
                'PROFESSIONAL': 3
            }

            df_unseen['EDUCATION'] = df_unseen['EDUCATION'].map(education_map).astype(int)
            df_unseen_encoded = pd.get_dummies(df_unseen, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

            missing_cols = set(x.columns) - set(df_unseen_encoded.columns)

            for col in missing_cols:
                df_unseen_encoded[col] = 0

            df_unseen_encoded = df_unseen_encoded[x.columns]

            y_pred_unseen = xgb_classifier.predict(df_unseen_encoded)

            df_unseen['Predicted_Target'] = label_encoder.inverse_transform(y_pred_unseen)

            st.write("Unseen Data Predictions")
            st.write(df_unseen)

            xlsx_file = "predicted_dataset.xlsx"

            with pd.ExcelWriter(xlsx_file, engine='xlsxwriter') as writer:
                df_unseen.to_excel(writer, index=False, sheet_name='Predictions')

            with open(xlsx_file, "rb") as f:
                st.download_button(
                    label="Download Predicted Dataset",
                    data=f,
                    file_name=xlsx_file,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            end_time = time.time()

            elapsed_time = end_time - start_time
            st.write(f"Total run time of the program: {round(elapsed_time, 2)} sec")
    else:
        st.write("Please upload all three files to proceed.")

elif page == "EPS":
    model = pickle.load(open('EPS/eps_v1.sav', 'rb'))

    st.title("EPS Prediction")

    roce = st.number_input("ROCE (%):", min_value=0.0) # Return on Capital Employed
    casa = st.number_input("CASA (%):", min_value=0.0) # Current Account and Savings Account Ratio
    roe_networth = st.number_input("Return on Equity / Networth (%):", min_value=0.0) # Return on Equity
    non_interest_income = st.number_input("Non-Interest Income/Total Assets (%):", min_value=0.0) # Non-Interest Income as a Percentage of Total Assets
    operating_profit = st.number_input("Operating Profit/Total Assets (%):", min_value=0.0) # Operating Profit as a Percentage of Total Assets
    operating_expenses = st.number_input("Operating Expenses/Total Assets (%):", min_value=0.0) # Operating Expenses as a Percentage of Total Assets
    interest_expenses = st.number_input("Interest Expenses/Total Assets (%):", min_value=0.0) # Interest Expenses as a Percentage of Total Assets
    face_value = st.number_input("Face value:", min_value=0.0) # Face Value of Shares

    if st.button("Predict using ML"):
        input_data = [[roce, casa, roe_networth, non_interest_income, operating_profit, operating_expenses, interest_expenses, face_value]]

        result = model.predict(input_data)[0]
        
        st.write(f"**EPS Predicted:** {result}")

elif page == "About":
    st.title("About")

    st.write("### Credit Risk Modeling")
    st.write("We developed the credit risk modeling framework to assess the likelihood of borrowers defaulting on loans. By employing statistical and machine learning methods, we analyze historical data to identify patterns that can predict creditworthiness. We examine factors such as age, marital status, education, and financial history to build robust models. This enables financial institutions to evaluate lending risks effectively and make informed decisions, ultimately improving their risk management strategies.")

    st.write("### Earnings Per Share (EPS)")
    st.write("We built this EPS prediction model to provide insights into a company's profitability on a per-share basis. By utilizing machine learning techniques, we analyze various financial indicators such as Return on Capital Employed (ROCE), Return on Equity (ROE), and operating expenses. Our goal is to help stakeholders make informed investment decisions by accurately predicting EPS based on historical data and financial metrics.")

    st.write("### Team Members")
    st.write("- **Urvi Chawla**: E22CSEU1247")
    st.write("- **Aryan Niranjan**: E22CSEU1250")
    st.write("- **Mehul Pathak**: E22CSEU1253")
    st.write("- **Varun Pathak**: E22CSEU1257")
    st.write("- **Prabhav Khanduri**: E22CSEU1759")