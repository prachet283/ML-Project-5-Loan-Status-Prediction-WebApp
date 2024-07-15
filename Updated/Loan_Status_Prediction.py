# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:01:27 2024

@author: prachet
"""

import json
import pickle
import streamlit as st
import pandas as pd

#loading. the saved model
with open("Updated/columns.pkl", 'rb') as f:
    all_columns = pickle.load(f)
with open("Updated/cat_columns.pkl", 'rb') as f:
    cat_columns = pickle.load(f)
with open("Updated/encoder.pkl", 'rb') as f:
    encoder = pickle.load(f)
with open("Updated/encoded_columns.pkl", 'rb') as f:
    encoded_columns = pickle.load(f)
with open("Updated/training_columns.pkl", 'rb') as f:
    training_columns = pickle.load(f)
with open("Updated/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
with open("Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb = json.load(file)
with open("Updated/best_features_rfc.json", 'r') as file:
    best_features_rfc = json.load(file)
with open("Updated/best_features_svc.json", 'r') as file:
    best_features_svc = json.load(file)
with open("Updated/loan_status_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb = pickle.load(f)
with open("Updated/loan_status_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc = pickle.load(f)
with open("Updated/loan_status_trained_svc_model.sav", 'rb') as f:
    loaded_model_svc = pickle.load(f)


#creating a function for prediction

def loan_status_prediction(input_data):

    #loading columns
    columns = all_columns
    
    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=columns)
    
    # Convert the categorical columns to string type
    df[cat_columns] = df[cat_columns].astype('str')
    
    # Encode the categorical columns
    input_data_encoded = encoder.transform(df[cat_columns])
    
    # Create a DataFrame with the encoded features
    input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns)
    
    # Add the remaining non-categorical columns
    input_data_final_encoded = pd.concat([df.drop(cat_columns, axis=1).reset_index(drop=True), input_data_encoded_df], axis=1)
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data_final_encoded)
    
    # Create a DataFrame with the standardized features
    input_data_df = pd.DataFrame(input_data_scaled, columns=training_columns)
    
    #loading best features
    df_best_features_xgb = input_data_df[best_features_xgb]
    df_best_features_rfc = input_data_df[best_features_rfc]
    df_best_features_svc = input_data_df[best_features_svc]
    
    #predictions
    prediction1 = loaded_model_xgb.predict(df_best_features_xgb)
    prediction2 = loaded_model_rfc.predict(df_best_features_rfc)
    prediction3 = loaded_model_svc.predict(df_best_features_svc)
    
    return prediction1 , prediction2, prediction3
  
    
  
def main():
    
    #giving a title
    st.title('Loan Status Prediction using ML')
    
    col1 , col2 = st.columns(2)
    #getting input data from user
    with col1:
        no_of_dependents = st.number_input("No of Dependents")
    with col2:
        education = st.selectbox('Education',('Graduate', 'Not Graduate')) 
    with col1:
        self_employed = st.selectbox('Self_Employed',('Yes', 'No')) 
    with col2:
        income_annum = st.number_input("Income Annum")
    with col1:
        loan_amount = st.number_input("Loan Amount")
    with col2:
        loan_term = st.number_input("Loan Term")
    with col1:
        cibil_score = st.number_input('Cibil Score')
    with col2:
        residential_assets_value = st.number_input('Residential Assets Value')
    with col1:
        commercial_assets_value = st.number_input('Commercial Assets Value')
    with col2:
        luxury_assets_value = st.number_input('Luxury Assets Value')
    with col1:
        bank_asset_value = st.number_input('Bank Asset Value')
        
        
    # code for prediction
    loan_status_xgb = ''
    loan_status_rfc = ''
    loan_status_svc = ''
    
    loan_status_xgb,loan_status_rfc,loan_status_svc = loan_status_prediction([no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value])
    
    
    #creating a button for Prediction
    if st.button("Predict Loan Status"):
        if(loan_status_xgb[0]==0):
            prediction = 'The Loan of the Person is Accepted' 
        else:
            prediction = 'The Loan of the Person is Rejected'
        st.write(f"Prediction: {prediction}")
    
    if st.checkbox("Show Advanced Options"):
        if st.button("Predict Loan Status with XG Boost Classifier"):
            if(loan_status_xgb[0]==0):
                prediction = 'The Loan of the Person is Accepted' 
            else:
                prediction = 'The Loan of the Person is Rejected'
            st.write(f"Prediction: {prediction}")
        if st.button("Predict Loan Status with Random Forest Classifier"):
            if(loan_status_rfc[0]==0):
                prediction = 'The Loan of the Person is Accepted' 
            else:
                prediction = 'The Loan of the Person is Rejected'
            st.write(f"Prediction: {prediction}")
        if st.button("Predict Loan Status with Support Vector Classifier"):
            if(loan_status_svc[0]==0):
                prediction = 'The Loan of the Person is Accepted' 
            else:
                prediction = 'The Loan of the Person is Rejected'
            st.write(f"Prediction: {prediction}")  
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
