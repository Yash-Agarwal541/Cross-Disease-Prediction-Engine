# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:52:41 2025

@author: HP
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

#loading the saved model
diabetes_model = pickle.load(open('C:/Users/HP/OneDrive/Desktop/Multipal_disease_prediction/multipal_medical_model/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/HP/OneDrive/Desktop/Multipal_disease_prediction/multipal_medical_model/heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/HP/OneDrive/Desktop/Multipal_disease_prediction/multipal_medical_model/parkinsons_model.sav', 'rb'))
breast_model = pickle.load(open('C:/Users/HP/OneDrive/Desktop/Multipal_disease_prediction/multipal_medical_model/breast_model.sav', 'rb'))
#loading scaler
scaler = pickle.load(open('C:/Users/HP/OneDrive/Desktop/Multipal_disease_prediction/scaler.sav', 'rb'))
#sidebar for navigation
with st.sidebar:
    selected = option_menu('Cross-Disease Prediction Engine',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Breast Cancer Prediction'],
                           icons = ['activity', 'clipboard2-heart', 'person-bounding-box', 'clipboard2-pulse'],
                           default_index = 0)
    
#Diabetes Prediction Page
if(selected == 'Diabetes Prediction'):
    #page title
    st.title('Diabetes Prediction using ML')
    
    # Getting the input data from the user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Value')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    #code for Prediction
    diab_dignosis = ''
    
    #creating Button for Prediction
    if st.button('Click To Get Diabetes Test Result'):
        #double square bracket is used because to tell model to predict for one value
        input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
        input_data = scaler.transform(input_data)
        diab_prediction = diabetes_model.predict(input_data)
        if(diab_prediction[0] == 1):
            diab_dignosis = 'The Person Is Diabetic'
        else:
            diab_dignosis = 'The Person Is Not Diabetic'
    st.success(diab_dignosis)

if(selected == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('sex')
    with col3:
        cp = st.text_input('Chest Pain type')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Include Angina')
    with col1:
        oldpeak = st.text_input('ST Depression Include By Excercise')
    with col2:
        slope = st.text_input('Slope Of The Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('Major Vessels Colored By Flourosopy')
    with col1:
        thal = st.text_input('0 --> normal; 1 --> fixed defect 2 --> reversable defect')
    
    heart_diagnosis = ''
    
    if st.button('Click To Get Heart Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(i) for i in user_input]
        heart_diagnosis = heart_disease_model.predict([user_input])
        if heart_diagnosis[0] == 1:
            heart_diagnosis = 'The Person Is Having Heart Disease'
        else:
            heart_diagnosis = 'The Person Does Not Have Any Heart Disease'
    st.success(heart_diagnosis)
    
    
#Parkinson 
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
    
if(selected == 'Breast Cancer Prediction'):
    st.title('Breast Cancer Prediction using ML')
    
    col1, col2, col3, col4, col5 = st.columns(5)    
    with col1:
        mr = st.text_input('Mean radius')

    with col2:
        texture = st.text_input('Mean Texture')

    with col3:
        peri = st.text_input('Mean Perimeter')
        
    with col4:
        area = st.text_input('Mean Area')

    with col5:
        smooth = st.text_input('Mean Smoothness')

    with col1:
        compact = st.text_input('Mean Compactness')

    with col2:
        concavity = st.text_input('Mean Concavity')

    with col3:
        concave = st.text_input('Mean Concave Points')

    with col4:
        symmetry = st.text_input('Mean Symmetry')

    with col5:
        fd = st.text_input('Mean Fractal Dimension')

    with col1:
        re = st.text_input('Radius Error')

    with col2:
        te = st.text_input('Texture Error')

    with col3:
        pe = st.text_input('Perimeter Error')

    with col4:
        ae = st.text_input(' Area Error')

    with col5:
        se = st.text_input('Smoothness Error')

    with col1:
        ce = st.text_input('Compactness Error')

    with col2:
        cone = st.text_input('Concavity Error')

    with col3:
        conce = st.text_input('Concave Point Error')

    with col4:
        syse = st.text_input('Sysmmetry Error')

    with col5:
        fe = st.text_input('Fractional Dimension Error')

    with col1:
        wr = st.text_input('Worst Radius Error')
    with col2:
        wt = st.text_input('Worst Texture Error')

    with col3:
        wp = st.text_input('Worst Perimeter Error')

    with col4:
        wa = st.text_input('Worst Area')

    with col5:
        ws = st.text_input('Worst Smoothness')

    with col1:
        wc = st.text_input('Worst Compactness')
    with col2:
        wcon = st.text_input('Worst Concavity')

    with col3:
        wconc = st.text_input('Worst Concave Points')

    with col4:
        wsys = st.text_input('Worst Symmetry')

    with col5:
        wfd = st.text_input('Worst Fractal Dimension')
        
    Breast_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Breast Cancer Test Result"):

        user_input = [mr, texture, peri, area, smooth, compact, concavity, concave, symmetry, fd,
                      re, te, pe, ae, se, ce, cone, conce, syse, fe,
                      wr, wt, wp, wa, ws, wc, wcon, wconc, wsys, wfd]

        user_input = [float(x) for x in user_input]

        Breast_Cancer_Prediction = breast_model.predict([user_input])

        if Breast_Cancer_Prediction[0] == 1:
            Breast_diagnosis = "The person has Breast Cancer"
        else:
            Breast_diagnosis = "The person does not have Breast Cancer"

    st.success(Breast_diagnosis)
    
    
    
