import pandas as pd
import numpy as np
import requests 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load and preprocess data
df = pd.read_csv(r"\Users\hv795\Desktop\gwl-1\full_data.csv")  # Replace with your file path
df['precip'] = df['precip'].replace(0, None).bfill().infer_objects()
df['soil_moisture_proxy'] = df['precip'] * (df['humidity'] / 100) * (1 - df['solarradiation']/max(df['solarradiation']))
df['groundwater_level_proxy'] = df['precip'] * (df['humidity'] / 100) * (1 - df['solarradiation'] / max(df['solarradiation']))

df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df.drop(['datetime'], axis=1, inplace=True)


# Features and target
X = df[['temp', 'humidity', 'precip', 'windspeed', 'cloudcover', 'dew', 'solarradiation', 'soil_moisture_proxy', 'year']]
y = df['groundwater_level_proxy']
import matplotlib.pyplot as plt

# Plot the distribution of the target variable
plt.hist(y, bins=50)
plt.title("Distribution of Groundwater Level Proxy")
plt.xlabel("Groundwater Level Proxy")
plt.ylabel("Frequency")
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Before scaling (first row):", X_train.iloc[0])
print("After scaling (first row):", X_train_scaled[0])


# Initialize and train the Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
plt.scatter(y_test, y_pred_rf)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Groundwater Level Proxy")
plt.ylabel("Predicted Groundwater Level Proxy")
plt.show()


# Initialize and train the XGBoost Model
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))

# Train an ARIMA Model on the univariate time series data (y)
y_train_arima = y_train.values
arima_model = ARIMA(y_train_arima, order=(5, 1, 0))  # Adjust order based on analysis
arima_model_fit = arima_model.fit()
y_pred_arima = arima_model_fit.forecast(steps=len(y_test))
print("ARIMA R2:", r2_score(y_test, y_pred_arima))

# Prepare data for the LSTM model
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Initialize and train the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
print("LSTM R2:", r2_score(y_test, y_pred_lstm))

# Print R2 scores for all models
print("\nModel Performance Comparison (R2 Scores):")
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))
print("ARIMA R2:", r2_score(y_test, y_pred_arima))
print("LSTM R2:", r2_score(y_test, y_pred_lstm))

#weather data function
def get_weather_data(lat, lon): 
    api_key = 'f734ec4cc4d1440c835151610242910'  # Replace with your API key
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}"
    response = requests.get(url).json()
    print(response)
    temp = response['current']['temp_c']
    humidity = response['current']['humidity']
    precipitation = response['current']['precip_mm']
    wind = response['current']['wind_mph']
    cloud = response['current']['cloud']
    dewpoint = response['current']['dewpoint_c']

    return {
        'temp': temp,
        'humidity': humidity,
        'precip': precipitation,
        'wind': wind,
        'cloud': cloud,
        'dewpoint': dewpoint
    }

# Calculate soil moisture
def calculate_soil_moisture(precip, humidity, solar_radiation):
    soil_moisture_proxy = precip * (humidity / 100) * (1 - solar_radiation / max(1, solar_radiation))
    return soil_moisture_proxy

# Train solar radiation model
X1 = X.drop(['solarradiation', 'soil_moisture_proxy', 'year'], axis=1)
y = X['solarradiation']
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
solar_model = RandomForestRegressor(n_estimators=100)
solar_model.fit(X_train, y_train)
print("Solar Radiation Model R2:", r2_score(y_test, solar_model.predict(X_test)))


# Predict groundwater level function
def predict_groundwater_level(lat, lon):
    weather_data = get_weather_data(lat, lon)
    if weather_data is None:
        print("Error: Weather data could not be retrieved.")
        return None
    print("Weather Data Retrieved:", weather_data)
    solar_radiation = solar_model.predict([[weather_data['temp'], 
                                            weather_data['humidity'], 
                                            weather_data['precip'], 
                                            weather_data['wind'],
                                            weather_data['cloud'],
                                            weather_data['dewpoint']]])[0]

    soil_moisture_proxy = calculate_soil_moisture(
        weather_data['precip'], 
        weather_data['humidity'], 
        solar_radiation
    )

    features = np.array([[weather_data['temp'], 
                          weather_data['humidity'], 
                          weather_data['precip'], 
                          weather_data['wind'],
                          weather_data['cloud'],
                          weather_data['dewpoint'],
                          solar_radiation,
                          soil_moisture_proxy,
                          2024]]).reshape(1, -1)

    scaled_features = scaler.transform(features)
    y_pred = rf.predict(scaled_features)[0]

    y_pred_str = str(y_pred)[:6] 
    y_pred = float(y_pred_str) 

    if y_pred < 1:
        return y_pred, "EXCESS"
    elif y_pred > 1 and y_pred < 3:
        return y_pred, "NORMAL"
    elif y_pred > 5:
        return y_pred, "LOW"
    else:
        return y_pred, "MODERATE"
print(df[['soil_moisture_proxy', 'groundwater_level_proxy']].describe())

# User input for latitude and longitude
lat = float(input("Enter latitude: "))
lon = float(input("Enter longitude: "))

# Predict groundwater level
groundwater_level, condition = predict_groundwater_level(lat, lon)
print(f"Predicted groundwater level: {groundwater_level*100}, Condition: {condition}")
