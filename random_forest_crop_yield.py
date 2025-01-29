# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
data = pd.DataFrame({
    "Farm_ID": ["F001", "F002", "F003", "F004", "F005", "F006", "F007", "F008", "F009", "F010",
                "F011", "F012", "F013", "F014", "F015", "F016", "F017", "F018", "F019", "F020"],
    "Crop_Type": ["Cotton", "Carrot", "Sugarcane", "Tomato", "Tomato", "Sugarcane", "Soybean", "Rice", "Maize", "Soybean",
                  "Rice", "Sugarcane", "Wheat", "Rice", "Sugarcane", "Barley", "Carrot", "Maize", "Maize", "Barley"],
    "Farm_Area(acres)": [329.4, 18.67, 306.03, 380.21, 135.56, 12.5, 360.06, 464.6, 389.37, 184.37,
                         279.95, 145.32, 329.1, 246.02, 305.15, 60.22, 284.01, 128.23, 460.93, 58.85],
    "Irrigation_Type": ["Sprinkler", "Manual", "Flood", "Rain-fed", "Sprinkler", "Sprinkler", "Drip", "Drip", "Drip", "Drip",
                        "Drip", "Flood", "Drip", "Flood", "Rain-fed", "Flood", "Manual", "Rain-fed", "Drip", "Sprinkler"],
    "Fertilizer_Used(tons)": [8.14, 4.77, 2.91, 3.32, 8.33, 6.42, 1.83, 5.18, 0.57, 2.18,
                              8.02, 3.01, 5.26, 1.01, 5.39, 2.19, 5.89, 4.91, 1.09, 3.61],
    "Pesticide_Used(kg)": [2.21, 4.36, 0.56, 4.35, 4.48, 2.25, 2.37, 0.91, 4.93, 2.67,
                           1.24, 2.27, 0.83, 3.45, 2.15, 0.35, 0.81, 0.77, 1.31, 3.32],
    "Yield(tons)": [14.44, 42.91, 33.44, 34.08, 43.28, 38.18, 44.93, 4.23, 3.86, 17.25,
                    32.85, 8.08, 5.44, 11.38, 28.77, 16.03, 47.7, 16.67, 39.96, 18.85],
    "Soil_Type": ["Loamy", "Peaty", "Silty", "Silty", "Clay", "Loamy", "Sandy", "Silty", "Peaty", "Sandy",
                  "Clay", "Clay", "Clay", "Sandy", "Peaty", "Sandy", "Loamy", "Loamy", "Sandy", "Sandy"],
    "Season": ["Kharif", "Kharif", "Kharif", "Zaid", "Zaid", "Zaid", "Rabi", "Kharif", "Rabi", "Kharif",
               "Zaid", "Kharif", "Zaid", "Rabi", "Kharif", "Zaid", "Zaid", "Rabi", "Zaid", "Kharif"],
    "Water_Usage(cubic meters)": [76648.2, 68725.54, 75538.56, 45401.23, 93718.69, 46487.98, 40583.57, 9392.38, 60202.14, 90922.15,
                                  5869.75, 88976.51, 45922.35, 71953.14, 33615.77, 25132.48, 88301.46, 18660.03, 54314.28, 92481.89]
})

# 2. Data Preprocessing
# Encoding categorical columns
data = pd.get_dummies(data, columns=["Crop_Type", "Irrigation_Type", "Soil_Type", "Season"], drop_first=True)

# Splitting the data into training and test sets
X = data.drop(["Farm_ID", "Yield(tons)"], axis=1)
y = data["Yield(tons)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Model Training and Prediction
# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# 4. Model Evaluation
# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 5. Comparison: Actual vs Predicted
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Yield", marker='o')
plt.plot(y_pred, label="Predicted Yield", marker='x')
plt.xlabel("Sample")
plt.ylabel("Yield (tons)")
plt.title("Actual vs Predicted Yield")
plt.legend()
plt.show()
