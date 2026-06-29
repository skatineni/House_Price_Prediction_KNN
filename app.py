import streamlit as st
import pandas as pd
import joblib

# ------------------------------------
# Page Configuration
# ------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------------
# Load Dataset
# ------------------------------------

df = pd.read_csv("house_price_india_25000.csv")

# ------------------------------------
# Load Model
# ------------------------------------

model = joblib.load("house_price_knn.pkl")

# ------------------------------------
# Title
# ------------------------------------

st.title("🏠 House Price Prediction using KNN Regression")

st.write(
    "Predict the market price of a residential property using Machine Learning."
)

st.divider()

# ------------------------------------
# Sidebar Inputs
# ------------------------------------

st.sidebar.header("Property Details")

city = st.sidebar.selectbox(
    "City",
    sorted(df.City.unique())
)

locality = st.sidebar.selectbox(
    "Locality",
    sorted(df[df.City == city].Locality.unique())
)

property_type = st.sidebar.selectbox(
    "Property Type",
    sorted(df.Property_Type.unique())
)

bhk = st.sidebar.slider(
    "BHK",
    1,
    6,
    2
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1,
    6,
    2
)

builtup = st.sidebar.number_input(
    "Super Built-up Area (sqft)",
    500,
    5000,
    1200
)

carpet = st.sidebar.number_input(
    "Carpet Area (sqft)",
    400,
    4500,
    1000
)

floor = st.sidebar.number_input(
    "Floor",
    0,
    40,
    5
)

total_floors = st.sidebar.number_input(
    "Total Floors",
    1,
    50,
    20
)

property_age = st.sidebar.slider(
    "Property Age",
    0,
    30,
    5
)

furnishing = st.sidebar.selectbox(
    "Furnishing",
    sorted(df.Furnishing.unique())
)

parking = st.sidebar.slider(
    "Parking",
    0,
    4,
    1
)

metro = st.sidebar.slider(
    "Distance to Metro (km)",
    0.1,
    20.0,
    2.0
)

schools = st.sidebar.slider(
    "Nearby Schools",
    1,
    10,
    5
)

hospitals = st.sidebar.slider(
    "Nearby Hospitals",
    1,
    10,
    5
)

amenities = st.sidebar.slider(
    "Amenities Score",
    1,
    10,
    7
)

# ------------------------------------
# Predict Button
# ------------------------------------

if st.button("Predict House Price"):

    sample = pd.DataFrame({

        "Property_ID":[99999],

        "City":[city],

        "Locality":[locality],

        "Property_Type":[property_type],

        "BHK":[bhk],

        "Bathrooms":[bathrooms],

        "Super_Builtup_Area_sqft":[builtup],

        "Carpet_Area_sqft":[carpet],

        "Floor":[floor],

        "Total_Floors":[total_floors],

        "Property_Age":[property_age],

        "Furnishing":[furnishing],

        "Parking":[parking],

        "Distance_to_Metro_km":[metro],

        "Nearby_Schools":[schools],

        "Nearby_Hospitals":[hospitals],

        "Amenities_Score":[amenities]

    })

    price = model.predict(sample)[0]

    st.success(f"Estimated House Price : ₹ {price:,.0f}")

    st.balloons()