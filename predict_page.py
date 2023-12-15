import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('svr_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]



def show_predict_page():
    st.title("SPI Prediction")

    st.write("""### We need some information""")
    
    
    feature1 = st.number_input('submittal closure rate', value=0.0)
    feature2 = st.number_input('machinery', value=0.0)
    feature3 = st.number_input('IFF Issuance rate', value=0.0)
    feature4 = st.number_input('manpower', value=0.0)
    feature5 = st.number_input('fabrication variance', value=0.0)
    feature6 = st.number_input('safety incident', value=0.0)
    feature7 = st.number_input('rfi closure rate', value=0.0)
    feature8 = st.number_input('material PO issuance rate', value=0.0)
    

    ok = st.button("Predict")
    if ok:
        x = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 ]])
        x = x.astype(float)
        
    

       
        prediction = regressor.predict(x) 
        st.subheader(f"The estimated SPI value is {prediction[0]:.2f}")
    
show_predict_page()