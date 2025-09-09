import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # --- Load config.yaml ---
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # --- Initialize authenticator ---
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # --- Login UI ---
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        authenticator.logout("Logout", "sidebar")
        st.sidebar.success(f"Welcome {name} 👋")

        # 🔽 Paste your entire dashboard logic here:
        # File uploader, preprocessing, model training, plotting, metrics, download button, etc.

    elif authentication_status == False:
        st.error("Invalid username or password")
        st.stop()
    elif authentication_status == None:
        st.warning("Please enter your credentials")
        st.stop()
if __name__ == "__main__":
    main()

# Set page configuration
st.set_page_config(page_title="NIFTY-50 Forecasting", layout="wide")
st.title("📈 NIFTY-50 Forecasting with LSTM")
st.write("This app uses a Long Short-Term Memory (LSTM) network to forecast the NIFTY-50 index based on historical data.")

# --- Helper Functions ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Loads and preprocesses the data, including feature engineering."""
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns or 'Close' not in df.columns:
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Create synthetic index and features
    df_index = df.groupby('Date')['Close'].mean().reset_index()
    df_index.rename(columns={'Close': 'Index_Close'}, inplace=True)
    df_index['Daily_Return'] = df_index['Index_Close'].pct_change()
    df_index['MA_7'] = df_index['Index_Close'].rolling(window=7).mean()
    df_index['MA_30'] = df_index['Index_Close'].rolling(window=30).mean()
    df_index.set_index('Date', inplace=True)
    df_index.dropna(inplace=True)
    return df_index

def create_sequences(data, seq_length, target_col_idx):
    """Creates sequences of data for LSTM model."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, target_col_idx]) # Target is a specific column
    return np.array(X), np.array(y)

@st.cache_resource
def train_model(_X_train, _y_train):
    """Builds, compiles, and trains the LSTM model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(_X_train.shape[1], _X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(_X_train, _y_train, epochs=10, batch_size=32, verbose=0) # verbose=0 for cleaner Streamlit output
    return model

def plot_forecast(dates, actual, predictions):
    """Plots the actual vs. forecasted values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, actual, label='Actual Price', color='blue')
    ax.plot(dates, predictions, label='LSTM Forecast', color='orange', linestyle='--')
    ax.set_title('NIFTY-50 Forecast vs. Actual Price', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Index Price')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# --- Main App Logic ---

def main():
    uploaded_file = st.file_uploader("Upload your historical dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_processed = load_and_preprocess_data(uploaded_file)

        if df_processed is None:
            st.error("❌ The CSV must contain 'Date' and 'Close' columns.")
            return

        st.success("✅ File uploaded and processed successfully!")
        st.write("📄 Data Preview with Engineered Features:")
        st.dataframe(df_processed.head())

        # --- Data Preparation for LSTM ---
        features = ['Index_Close', 'Daily_Return', 'MA_7', 'MA_30']
        target_col = 'Index_Close'
        target_col_idx = features.index(target_col)

        data = df_processed[features].values

        # Split data before scaling to prevent data leakage
        split_ratio = 0.8
        split_idx = int(len(data) * split_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)

        # Create sequences
        seq_length = 60
        X_train, y_train = create_sequences(scaled_train_data, seq_length, target_col_idx)
        X_test, y_test = create_sequences(scaled_test_data, seq_length, target_col_idx)

        if len(X_train) == 0 or len(X_test) == 0:
            st.warning("⚠️ Not enough data to create training/testing sequences. Please upload a larger dataset.")
            return

        # --- Model Training and Prediction ---
        with st.spinner("Training LSTM model... This may take a moment on first run."):
            model = train_model(X_train, y_train)
        st.success("✅ Model trained successfully!")

        predictions_scaled = model.predict(X_test)

        # Inverse transform predictions
        # Create a dummy array with the same number of features as the scaler
        dummy_predictions = np.zeros((len(predictions_scaled), len(features)))
        dummy_predictions[:, target_col_idx] = predictions_scaled.ravel()
        predictions = scaler.inverse_transform(dummy_predictions)[:, target_col_idx]

        # Inverse transform actual values for comparison
        dummy_y_test = np.zeros((len(y_test), len(features)))
        dummy_y_test[:, target_col_idx] = y_test.ravel()
        actual = scaler.inverse_transform(dummy_y_test)[:, target_col_idx]

        # --- Display Results ---
        st.subheader("Forecast Results")
        
        # Get the dates for the test set for plotting
        test_dates = df_processed.index[split_idx + seq_length:]

        fig = plot_forecast(test_dates, actual, predictions)
        st.pyplot(fig)

        # Display last few predictions in a table
        results_df = pd.DataFrame({
            'Date': test_dates,
            'Actual Price': actual,
            'Forecasted Price': predictions
        })
        st.write("Forecast vs. Actual (Last 10 days of test set):")
        st.dataframe(results_df.tail(10).set_index('Date'))

        # Calculate and display metrics
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        r2 = r2_score(actual, predictions)
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")
        # Download button for results
        st.markdown("### Download Forecast Results")
        st.write("You can download the forecast results as a CSV file for further analysis.")
        
        # Download button
        st.download_button(
        label="Download Forecast Results as CSV",
        data=results_df.to_csv(index=False).encode('utf-8'),
        file_name='nifty50_forecast.csv',
        mime='text/csv'
)    
        retrain = st.checkbox("Retrain model from scratch", value=False)
        if retrain:
            model = train_model(X_train, y_train)
        else:
            model = train_model(X_train, y_train)  # Cached version        
        
        

    else:
        st.info("ℹ️ Please upload a CSV file to begin forecasting.")

if __name__ == "__main__":
    main()
