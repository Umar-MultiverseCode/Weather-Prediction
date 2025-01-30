# Importing necessary libraries
import pandas as pd                    # For data manipulation and analysis
from sklearn.linear_model import LogisticRegression  # For building the logistic regression model
from sklearn.preprocessing import StandardScaler      # For feature scaling (normalization)
import streamlit as st                 # For building the interactive web app


# Convert to DataFrame - converting the above data dictionary to a pandas DataFrame
df = pd.read_csv('weather_data_2012.csv')

# Features (X) and target (y)
X = df[['Temperature', 'Humidity', 'Pressure']]  # Features (independent variables)
y = df['Rain']  # Target variable (dependent variable)

# Train the model
scaler = StandardScaler()  # Initialize the StandardScaler to normalize the data
X_scaled = scaler.fit_transform(X)  # Scale the features (standardize them)

# Initialize Logistic Regression model
model = LogisticRegression()  
model.fit(X_scaled, y)  # Train the model with the scaled features and target

# Function to predict rain based on user input
def predict_rain(temperature, humidity, pressure):
    # Take the user input (temperature, humidity, pressure) and create a DataFrame
    user_input = pd.DataFrame([[temperature, humidity, pressure]], columns=['Temperature', 'Humidity', 'Pressure'])
    
    # Scale the user input the same way as training data
    user_input_scaled = scaler.transform(user_input)

    # Predict rain (1 = Rain, 0 = No Rain)
    prediction = model.predict(user_input_scaled)  # Get the prediction from the model

    # Return the result as a message based on the prediction
    if prediction == 1:
        return "🌧 Rain is likely! 🌧"  # If rain is predicted
    else:
        return "☀ Rain is not likely. ☀"  # If no rain is predicted

# Streamlit UI to take user input and predict
# Streamlit title (this is displayed as the heading on the app)
st.title('🌦 Weather Prediction Model 🌦')

# Description - Brief description of what the app does
st.write("""
This model predicts whether it will rain or not based on the following weather parameters:
- Temperature (°C) , Humidity (%) , Pressure (hPa)

Enter the values below and click "Predict Rain" to see the result!
""")

# Get user inputs (temperature, humidity, and pressure) using number input widgets
temperature = st.number_input("Enter Temperature (°C):", min_value=-50.0, max_value=50.0, value=25.0, step=0.1)  # Temperature input
humidity = st.number_input("Enter Humidity (%):", min_value=0, max_value=100, value=50, step=1)  # Humidity input
pressure = st.number_input("Enter Pressure (hPa):", min_value=900, max_value=1050, value=1015, step=1)  # Pressure input

# Predict when button is clicked
if st.button('🔮 Predict Rain 🔮'):  # When the "Predict Rain" button is clicked
    result = predict_rain(temperature, humidity, pressure)  # Call the function to predict rain
    # Display the result with different styles based on prediction
    st.markdown(f"### Prediction Result:")  # Display the title for the result
    st.markdown(f"#### {result}")  # Display the result (rain or no rain) based on prediction
