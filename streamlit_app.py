pip install matplotlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("NIFTY-50 Forecasting with LSTM")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("synthetic_nifty.csv")  # Your default dataset

# Show raw data
st.subheader("Raw Data")
st.dataframe(df.head())

# Show actual vs predicted plot
st.subheader("Forecast Plot")
fig, ax = plt.subplots()
ax.plot(y_test_actual, label='Actual')
ax.plot(y_pred_actual, label='Predicted')
ax.legend()
st.pyplot(fig)

# Show metrics
st.subheader("Model Performance")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")
st.write(f"**R² Score:** {r2:.2f}")
