import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('Product_Pricing_Dataset.csv')

# Preprocessing
X = df[["Price", "Competitor_Price", "Price_Elasticity", "Customer_Rating"]]
y = df["Demand"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Product Pricing Optimization")

# Display the dataset
st.subheader("Dataset")
st.dataframe(df)

# Visualization of aggregated demand by region
st.subheader("Demand by Region")
st.bar_chart(df.groupby("Region")["Demand"].sum())

# Inputs for prediction
st.subheader("Predict Demand")
price = st.number_input("Enter Product Price:", min_value=0.0, max_value=500.0, value=100.0)
competitor_price = st.number_input("Enter Competitor Price:", min_value=0.0, max_value=500.0, value=100.0)
price_elasticity = st.number_input("Enter Price Elasticity:", min_value=-2.0, max_value=0.0, value=-1.0)
customer_rating = st.number_input("Enter Customer Rating:", min_value=1.0, max_value=5.0, value=3.0)

# Prediction
input_data = np.array([[price, competitor_price, price_elasticity, customer_rating]])
predicted_demand = model.predict(input_data)[0]

st.write(f"### Predicted Demand: {predicted_demand:.2f} units")
