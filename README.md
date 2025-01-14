 
# 🌦️ **Weather Prediction Model** 🌦️

Welcome to the **Weather Prediction Model**! 🌍 This model uses temperature, humidity, and pressure data to predict whether it will rain or not. The prediction is based on historical weather data and is powered by **Logistic Regression**. You can enter the values for temperature, humidity, and pressure, and the model will predict whether rain is likely or not! 🌧️



## 🔍 **What is this model?**

This model is designed to predict **rainfall** based on the following weather parameters:
- **Temperature** (°C)
- **Humidity** (%)
- **Pressure** (hPa)

Using these inputs, the model tells us whether **rain** will happen or not. 💧



## ⚙️ **How Does it Work?**

1. **Data Collection**: We train the model with sample data that includes temperature, humidity, and pressure, and the corresponding rain data (Rain: 1, No Rain: 0).
2. **Logistic Regression Model**: The core of the prediction is a **Logistic Regression** model. It analyzes the relationship between the weather parameters and rainfall.
3. **User Input**: You enter the values for temperature, humidity, and pressure.
4. **Prediction**: The model processes your input and predicts whether it will rain or not.



## 🚀 **How to Use It?**

### 1. 🧑‍💻 **Run the Streamlit App**

To run the app and make predictions:

1. **Install the required dependencies**:
    - If you haven’t already installed Streamlit, run the following:
      ```bash
      pip install streamlit
      ```

2. **Run the App**:
    - In your terminal, go to the folder where the code is saved and run:
      ```bash
      streamlit run weather_prediction_model.py
      ```
    - This will open the app in your default web browser.

### 2. 💡 **Input Values**

- **Enter Temperature (°C)**: The current temperature in degrees Celsius. 
- **Enter Humidity (%)**: The percentage of humidity in the air.
- **Enter Pressure (hPa)**: The atmospheric pressure in hectopascals.

### 3. 🔮 **Get the Prediction**

- Click the **"Predict Rain"** button to see the prediction:  
   - **🌧️ "Rain is likely!"**  
   - **☀️ "Rain is not likely."**



## 📊 **Example Prediction**

| Temperature (°C) | Humidity (%) | Pressure (hPa) | Predicted Rain |
|------------------|--------------|-----------------|----------------|
| 25               | 80           | 1015            | 🌧️ Rain is likely!  |
| 28               | 70           | 1010            | ☀️ Rain is not likely.  |



## 🔧 **Technologies Used**

- **Streamlit**: For building the interactive web app.
- **Logistic Regression**: A machine learning algorithm used to predict the likelihood of rain.
- **Scikit-learn**: For implementing the machine learning model and scaling the input data.



## 🎨 **How the Prediction Looks**

Here’s how your prediction looks when you input data:

1. **Title**: "🌦️ **Weather Prediction Model** 🌦️"
2. **User Inputs**: A nice user-friendly interface to input temperature, humidity, and pressure.
3. **Result**: After clicking the "🔮 **Predict Rain** 🔮" button, you’ll see either:
    - **🌧️ "Rain is likely!"** (with a rain emoji)
    - **☀️ "Rain is not likely."** (with a sun emoji)



## 📚 **Explanation of the Code**

- **Data**: We provide sample weather data (Temperature, Humidity, Pressure) and the target variable (Rain).
- **Scaling**: The input data is scaled using **StandardScaler** to ensure consistency and improve model performance.
- **Model**: We use **Logistic Regression** to train the model and predict the likelihood of rain based on the user input.
- **Streamlit**: The user interface is built using **Streamlit** to make it easy to interact with the model and get predictions.



## 💻 **Installation Instructions**

1. Clone or download the repository.
2. Navigate to the project directory.
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run weather_prediction_model.py
    ```



## 📝 **Future Improvements**

- 🌧️ **Use larger datasets**: Train the model with real-world weather datasets to improve accuracy.
- ⚙️ **Add more features**: Include additional weather features like wind speed, cloud cover, etc.
- 🌍 **Deploy on cloud**: Host the app online so that anyone can use it.



## 📬 **Feedback & Contact**

We’d love to hear your feedback! Feel free to open an issue or reach out to us.



### Enjoy using the **Weather Prediction Model**! 🌦️


 