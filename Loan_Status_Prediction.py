# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:01:27 2024

@author: prachet
"""

import numpy as np
import pickle
import streamlit as st

#loading. the saved model
loaded_model = pickle.load(open('loan_prediction_trained_model.sav','rb'))

#creating a function for prediction

def loan_status_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction)

    if(prediction[0]==0):
      return 'The person is not eligible for loan' 
    else:
      return 'The person is eligible'
  
    
  
def main():
    
    #giving a title
    st.title('Loan Status Prediction Web App')
    
    #getting input data from user
    
    col1 , col2 = st.columns(2)
    #getting input data from user
    with col1:
        option1 = st.selectbox('Gender',('Male', 'Female')) 
        gender = 1 if option1 == 'Female' else 0
    with col2:
        option2 = st.selectbox('Married',('Yes', 'No')) 
        married = 1 if option2 == 'Yes' else 0
    with col1:
        dependents = st.number_input("Dependents")
    with col2:
        option3 = st.selectbox('Education',('Graduate', 'Not Graduate')) 
        education = 1 if option3 == 'Graduate' else 0
    with col1:
        option4 = st.selectbox('Self_Employed',('Yes', 'No')) 
        self_employed = 1 if option4 == 'Yes' else 0
    with col2:
        applicant_income = st.number_input("Applicant Income")
    with col1:
        coapplicant_income = st.number_input("Coapplicant Income")
    with col2:
        loan_amount = st.number_input("Loan Amount")
    with col1:
        loan_amount_term = st.number_input('Loan Amount Term')
    with col2:
        credit_history = st.number_input('Credit History')
    with col1:
        option5 = st.selectbox('Property Area',('Rural', 'Urban','Semiurban'))
        if option5 == 'Rural':
            property_area = 0
        elif option5 == 'Urban':
            property_area = 1
        else:
            property_area = 2
   
    # code for prediction
    loan_status = ''
   
    #creating a button for Prediction
    if st.button('Predict Loan Status'):
       loan_status =loan_status_prediction((gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area)) 
    st.success(loan_status)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
