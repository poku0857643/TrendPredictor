import streamlit as st
import pandas as pd
from prophet import Prophet

st.title("Stock Price Trend")

# Load data
data = pd.read_csv('data.csv')

# Ensure the 'ds' column is properly converted to datetime format
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# If 'ds' contains strings or mixed types, it will convert invalid dates to NaT
# Drop any rows with invalid 'ds' values (NaT values)
data = data.dropna(subset=['date'])

# Convert 'y' to numeric, coercing errors to NaN
data['change'] = pd.to_numeric(data['change'], errors='coerce')

# Drop rows with NaN values in 'y' (in case there were invalid values)
data = data.dropna(subset=['change'])

# Prepare the data for Prophet
data['date'] = pd.to_datetime(data['date'])
data = data.rename(columns={'date': 'ds', 'change': 'y'})

# Drop rows with NaN values (if any)
data = data.dropna()

# Train the model
model = Prophet()
model.fit(data)

# Make future predictions
future = model.make_future_dataframe(30) # 30 days
forecast = model.predict(future)

# Plot predictions
st.write("Data Table", data)
fig = model.plot(forecast)
st.pyplot(fig)
st.feedback("stars")