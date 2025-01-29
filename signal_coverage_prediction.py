import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = r'C:\Users\acer\Downloads\Kondisi_Sinyal_Telepon_Seluler_Di_Wilayah_desa_cleaned.csv'
data = pd.read_csv(file_path)

# Check the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())
print("\nSummary statistics of the dataset:")
print(data.describe())

# Preprocessing: Convert columns with percentages to numerical values
columns_to_convert = ['PERSEN 2G', 'PERSEN 3G', 'PERSEN 4G', 'PERSEN 5G', 
                      'PERSEN MUK 2G', 'PERSEN MUK 3G', 'PERSEN MUK 4G', 'PERSEN MUK 5G',
                      'PER 2G', 'PER 3G', 'PER 4G', 'PER 5G', 
                      'PER MUK 2G', 'PER MUK 3G', 'PER MUK 4G', 'PER MUK 5G']

# Convert percentage columns to numeric
for col in columns_to_convert:
    data[col] = data[col].astype(str).str.replace('%', '').str.replace(',', '.')
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN values in the columns to convert
data.dropna(subset=columns_to_convert, inplace=True)

# Check for NaN values in the dataset after conversion
print("\nNaN values in each column after conversion:")
print(data.isnull().sum())

# Define independent variables (X) and dependent variable (y)
X = data.drop(columns=['PERSEN 4G'])
y = data['PERSEN 4G']

# Check the shape of X and y
print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# Check for non-numeric data in features and handle it
X = X.apply(pd.to_numeric, errors='coerce')
X.dropna(inplace=True)  # Drop rows with NaN values after conversion

# Ensure that the target variable (y) is aligned with the features (X)
y = y[X.index]  # Align y with the cleaned X

# Check for empty DataFrames
if X.empty or y.empty:
    print("No valid data available for training.")
    exit()  # Stop the execution if there's no valid data

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Future prediction (example)
new_data = {
    'PERSEN 2G': [50.0], 
    'PERSEN 3G': [40.0], 
    'PERSEN 5G': [10.0], 
    'PERSEN MUK 2G': [30.0], 
    'PERSEN MUK 3G': [25.0], 
    'PERSEN MUK 4G': [80.0], 
    'PERSEN MUK 5G': [10.0],
    'PER 2G': [75.0], 
    'PER 3G': [65.0]
}

# Convert new data to DataFrame
new_data_df = pd.DataFrame(new_data)

# Make predictions
prediction = model.predict(new_data_df)
print("Prediksi PERSEN 4G untuk data baru:", prediction[0])
