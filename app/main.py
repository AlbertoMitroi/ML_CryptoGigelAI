import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CryptoGigel AI",
    page_icon="💰",
    layout="centered"
)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_resources():
    # Paths to models
    model_path = 'models/crypto_price_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    # Check if files exist to avoid crashing
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run the notebooks first.")
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_resources()

# --- UI HEADER ---
st.title("🚀 CryptoGigel: Bitcoin Predictor")
st.markdown("""
Welcome to **CryptoGigel**. This Machine Learning app predicts the **Close Price of Bitcoin for TOMORROW** based on today's market data.
""")

st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🎛️ Input Parameters")
st.sidebar.write("Adjust the values based on today's market:")

def user_input_features():
    # We need inputs for all features used during training
    # Feature order: ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_7', 'SMA_30', 'Pct_Change', 'Volatility', 'RSI']
    
    # Default values are approx recent Bitcoin numbers to make it user-friendly
    open_price = st.sidebar.number_input("Open Price ($)", value=95000.0)
    high_price = st.sidebar.number_input("High Price ($)", value=96500.0)
    low_price  = st.sidebar.number_input("Low Price ($)", value=94000.0)
    close_price = st.sidebar.number_input("Close Price ($)", value=96000.0)
    volume = st.sidebar.number_input("Volume", value=35000000000.0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Technical Indicators")
    sma_7 = st.sidebar.number_input("SMA 7 Days", value=94000.0)
    sma_30 = st.sidebar.number_input("SMA 30 Days", value=90000.0)
    pct_change = st.sidebar.number_input("Pct Change (e.g., 0.02)", value=0.01)
    volatility = st.sidebar.number_input("Volatility (Std Dev)", value=1500.0)
    rsi = st.sidebar.slider("RSI (0-100)", min_value=0, max_value=100, value=65)

    # Create a dictionary for the dataframe
    data = {
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume,
        'SMA_7': sma_7,
        'SMA_30': sma_30,
        'Pct_Change': pct_change,
        'Volatility': volatility,
        'RSI': rsi
    }
    return pd.DataFrame(data, index=[0])

# Get input from user
input_df = user_input_features()

# --- MAIN SECTION ---
st.subheader("📊 Current Market Data")
st.dataframe(input_df)

# --- PREDICTION LOGIC ---
if st.button("🔮 Predict Next Day Price", type="primary"):
    if model is not None and scaler is not None:
        # 1. Scale the input using the loaded scaler
        scaled_input = scaler.transform(input_df)
        
        # 2. Make prediction
        prediction = model.predict(scaled_input)
        result = prediction[0]
        
        # 3. Display Result
        st.success(f"💰 Predicted Bitcoin Price for Tomorrow: **${result:,.2f}**")
        
        # 4. Compare with current close
        current_close = input_df['Close'].values[0]
        diff = result - current_close
        
        if diff > 0:
            st.metric(label="Expected Movement", value="📈 UP", delta=f"+${diff:,.2f}")
        else:
            st.metric(label="Expected Movement", value="📉 DOWN", delta=f"-${abs(diff):,.2f}")
            
    else:
        st.error("Model could not be loaded.")
        
# --- DOCUMENTATION SECTION ---
st.markdown("---")
with st.expander("📖 See the Project Documentation (README)"):
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
            
        st.markdown(readme_text)
    except FileNotFoundError:
        st.error("README.md file not found. Make sure you are running the application from the main project folder.")
        
# --- FOOTER ---
st.markdown("---")
st.caption("Built with Python, Scikit-Learn & Streamlit for ML Project 2025.")