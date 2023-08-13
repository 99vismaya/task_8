# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:04:26 2023

@author: Dell
"""

import streamlit as st
import numpy as np
import pickle

emp_perf_model_path1 = 'task8.pkl'
emp_perf_model1 = pickle.load(
    open(emp_perf_model_path1, 'rb'))

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html = True)

def main():
    st.title('Weather predition App')
    st.markdown('Just Enter the following details and we will predict the weather')
    b = st.selectbox("temperature",('cool','hot','mild'))
    if b == 'cool':
        b=0
    elif b == 'hot':
        b=1
    else:
        b=2
    c = st.selectbox("humidity",('high','normal'))
    if c == "high":
        c=0
    else:
        c=1
    d = st.selectbox("windy",("FALSE","TRUE"))  
    if d == "FALSE":
        d=0
    else:
        d=1
    submit = st.button('Predict Performance')
    st.text("")
    if submit: 
        prediction = emp_perf_model1.predict([[b,c,d]])
        st.write(prediction)
if __name__ == '__main__':
    main()
