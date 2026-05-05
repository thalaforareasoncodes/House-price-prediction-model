import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("🏠 House Price Predictor")

st.write("Enter house details below:")

# Load dataset safely
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Feature engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

df = df.dropna()

# Features
X = df[[
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]]

y = df["median_house_value"]

# Train model (cached so it doesn't retrain every time)
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# User Inputs
median_income = st.number_input("Median Income", 0.0, 15.0, 5.0)
age = st.number_input("House Age", 0, 100, 20)
rooms = st.number_input("Total Rooms", 1, 10000, 2000)
bedrooms = st.number_input("Total Bedrooms", 1, 5000, 400)
population = st.number_input("Population", 1, 10000, 1000)
households = st.number_input("Households", 1, 5000, 300)

# Derived features
rooms_per_household = rooms / households
bedrooms_per_room = bedrooms / rooms
population_per_household = population / households

# Prediction button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "median_income": [median_income],
        "housing_median_age": [age],
        "total_rooms": [rooms],
        "total_bedrooms": [bedrooms],
        "population": [population],
        "households": [households],
        "rooms_per_household": [rooms_per_household],
        "bedrooms_per_room": [bedrooms_per_room],
        "population_per_household": [population_per_household]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted House Price: ${prediction:,.2f}")
