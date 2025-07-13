import streamlit as st
import joblib
import numpy as np




# load the model
with open('models\simple_linear_regression.pkl','rb')as file:
    model=joblib.load(file)

# tilte
st.set_page_config(page_title="Salary Predictor")
st.title("Salary Predictor App")
st.subheader("Predict your salary based on your experience")



#Sidebar
st.sidebar.header("Enter Your Details")
experience =st.sidebar.slider("years of Experience",
                            min_value=0.0,max_value=20.0,step=0.5)



# Button to predict
if st.sidebar.button("Predict Salary"):
    #predict salary 
    salary = model.predict(np.array([[experience]]))[0]


    #display reult
    st.success(f"Predict Salary :Rs.{salary:,.2f}") # success is green color

    #Additional Info
    st.info("This Prediction is based on a simple Linear Regression")



    #footer
st.markdown('----------')
st.markdown("Made with using Streamlit")







