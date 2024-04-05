import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# TradingView Advanced Chart widget
st.markdown("<h1>EURUSD Price Prediction</h1>", unsafe_allow_html=True)
st.components.v1.html("""
    <div class="tradingview-widget-container" style="width: 100%; height: 600px;">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {
        "width": "100%",
        "height": 300,
        "symbol": "FOREXCOM:EURUSD",
        "interval": "60",
        "timezone": "Asia/Jakarta",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "calendar": false,
        "support_host": "https://www.tradingview.com"
        }
        </script>
    </div>
""", height=300)

st.components.v1.html("""
    <div class="tradingview-widget-container" style="width: 100%; height: 600px;">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
        {
        "width": "100%",
        "height": 300,
        "colorTheme": "light",
        "isTransparent": false,
        "locale": "en",
        "importanceFilter": "0,1",
        "countryFilter": "us,lv,it,gb,ua,ie,ch,se,hu,is,es,gr,de,sk,rs,fr,fi,ru,ro,eu,ee,pt,pl,dk,cz,no,nl,cy,be,lu,lt,at"
        }
        </script>
    </div>
""", height=300)

# Function to load data from uploaded file
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function for model training and evaluation
def train_and_evaluate(data):
    # Step 3: Split the data into training and testing sets
    X = data[['Open', 'High', 'Low', 'Close']]  # Features
    y = data['Close']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Step 6: Make predictions
    # Assuming you want to predict the next price based on the last row in the dataset
    last_row = data.iloc[[-1]]
    next_price = model.predict(last_row[['Open', 'High', 'Low', 'Close']])

    return mse, y_test, y_pred, next_price[0]

# Streamlit UI
def main():

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load data from uploaded file
        data = load_data(uploaded_file)
        
        # Display uploaded data in a table
        st.write("Uploaded Data:")
        st.write(data)

        # Train and evaluate the model
        mse, y_test, y_pred, next_price = train_and_evaluate(data)

        # Display evaluation results
        st.write("Mean Squared Error:", mse)
        st.write("Predicted Next Price:", next_price)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Plot actual vs. predicted prices
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label='Actual')
        ax.plot(y_pred, color='red', label='Predicted')
        ax.set_xlabel('Index')
        ax.set_ylabel('Close Price')
        ax.set_title('EURUSD Price Prediction')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()

