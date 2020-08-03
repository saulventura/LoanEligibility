# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:21:40 2020

@author: Saul
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle 
import shap


with open('loan_classifier.pickle','rb') as f:
    classifier = pickle.load(f)
    

def main():
    st.title('Loan Eligibility')
    
    st.sidebar.title('Customer Data')
    
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    options = ["Male", "Female"]
    Gender_rb = st.sidebar.radio("Gender", options, key="my_radio1")

   # Gender = st.text_input('Gender',2)
    options_yes_no = ["Yes", "No"]
    Married_rb = st.sidebar.radio("Married", options_yes_no, key="my_radio2")
     
    #Married = st.text_input('Married',1)
    
    Dependents = st.sidebar.text_input('Dependents',0)
    Education = st.sidebar.text_input('Education',1)
    Self_Employed = st.sidebar.text_input('Self_Employed',0)
    ApplicantIncome = st.sidebar.text_input('ApplicantIncome',0.070489)
    CoapplicantIncome = st.sidebar.text_input('CoapplicantIncome',0)
    LoanAmount = st.sidebar.text_input('LoanAmount',0.201158)
    Loan_Amount_Term = st.sidebar.text_input('Loan_Amount_Term',360.0)
    Credit_History = st.sidebar.text_input('Credit_History',1)
    Property_Area = st.sidebar.text_input('Property_Area',3)
    

    result = ""
    if st.button('Predict'): 
        
        if Gender_rb == "Male":
            Gender=2
        elif Gender_rb == "Female":
            Gender=1
            
        if Married_rb == "Yes":
            Married=1
        elif Married_rb == "No":
            Married=0       
            
            
        
        X = [[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,
                                      Loan_Amount_Term,Credit_History,Property_Area]]
            
        result = classifier.predict(X)
        #st.success(result[0])
        if result == 'Y':
            decision = 'Eligible for a Loan'
            st.success('The customer is {}'.format(decision))
        elif result == 'N':
            decision = 'Not Eligible for a Loan'
            st.success('The customer is {}'.format(decision))

    
        shap.initjs()

        data_for_prediction = pd.DataFrame(X, columns= ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome', 'LoanAmount',                       'Loan_Amount_Term','Credit_History','Property_Area'])


        # explain the model's predictions using SHAP values
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values( data_for_prediction )

        # visualize the first prediction's explaination
        shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction ,link='logit', matplotlib=True, figsize=(18,6))

        st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
        plt.clf()
    
    
if __name__== '__main__' :
    main()
    
    
    
    
    
    

