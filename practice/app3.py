import streamlit as st
import joblib
import numpy as np
import pandas as pd

# load the model
with open('models/MLR_3.pkl', 'rb') as f:
    model = joblib.load(f)

# title
st.set_page_config(page_title="House Price Predictor")
st.title("House Price Predictor App")
st.subheader("Predict your house price based on your area and requirements")

# Sidebar for user input
st.sidebar.header("Enter Your House Details")
bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0)
sqft_living = st.sidebar.number_input("Sqft Living", min_value=0, max_value=10000, value=1800)
sqft_lot = st.sidebar.number_input("Sqft Lot", min_value=0, max_value=100000, value=5000)
floors = st.sidebar.number_input("Floors", min_value=0.0, max_value=5.0, value=1.0)
waterfront = st.sidebar.selectbox("Waterfront", options=[0, 1], index=0)
view = st.sidebar.number_input("View", min_value=0, max_value=4, value=0)
condition = st.sidebar.number_input("Condition", min_value=1, max_value=5, value=3)
grade = st.sidebar.number_input("Grade", min_value=1, max_value=13, value=7)
sqft_above = st.sidebar.number_input("Sqft Above", min_value=0, max_value=10000, value=1500)
sqft_basement = st.sidebar.number_input("Sqft Basement", min_value=0, max_value=5000, value=300)
yr_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=1995)
yr_renovated = st.sidebar.number_input("Year Renovated", min_value=0, max_value=2025, value=0)
zipcode = st.sidebar.number_input("Zipcode", min_value=98001, max_value=98199, value=98001)
lat = st.sidebar.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.5112)
long = st.sidebar.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.257)
sqft_living15 = st.sidebar.number_input("Sqft Living 15", min_value=0, max_value=10000, value=1500)
sqft_lot15 = st.sidebar.number_input("Sqft Lot 15", min_value=0, max_value=100000, value=5000)

# Button to predict
if st.sidebar.button("Predict House Price"):
    input_dict = {
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'condition': [condition],
        'grade': [grade],
        'sqft_above': [sqft_above],
        'sqft_basement': [sqft_basement],
        'yr_built': [yr_built],
        'yr_renovated': [yr_renovated],
        'zipcode': [zipcode],
        'lat': [lat],
        'long': [long],
        'sqft_living15': [sqft_living15],
        'sqft_lot15': [sqft_lot15]
    }
    features_df = pd.DataFrame(input_dict)
    price = model.predict(features_df)[0]
    st.success(f"Predicted House Price: Rs.{price:,.2f}")
    st.info("This prediction is based on a Multiple Linear Regression model.")

# Footer
st.markdown('----------')
st.markdown("Made with Streamlit")







