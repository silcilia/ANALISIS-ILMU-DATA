import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1. Simulasi data historis berbasis waktu
np.random.seed(42)
n = 100  # Jumlah data historis
data = pd.DataFrame({
    'Tanggal': pd.date_range(start='2023-01-01', periods=n, freq='D'),
    'Nilai': np.cumsum(np.random.randn(n) * 10 + 50),  # Data historis dengan tren acak
    'Crop_Type': np.random.choice(['Cotton', 'Carrot', 'Sugarcane', 'Tomato'], size=n)
})

# 2. Tambahkan fitur waktu
data['Day'] = data['Tanggal'].dt.day
data['Month'] = data['Tanggal'].dt.month
data['Year'] = data['Tanggal'].dt.year

# 3. Split data untuk training dan forecasting
train = data[:-10]  # Gunakan sebagian besar data untuk pelatihan
forecast_dates = pd.date_range(start=data['Tanggal'].iloc[-1] + pd.Timedelta(days=1), periods=10, freq='D')
forecast_df = pd.DataFrame({
    'Tanggal': forecast_dates,
    'Day': forecast_dates.day,
    'Month': forecast_dates.month,
    'Year': forecast_dates.year,
    'Crop_Type': np.random.choice(['Cotton', 'Carrot', 'Sugarcane', 'Tomato'], size=10)  # Dummy kategori untuk prediksi
})

# Gabungkan data untuk encoding agar memiliki kolom yang sama
combined_data = pd.concat([train, forecast_df], ignore_index=True)
combined_data = pd.get_dummies(combined_data, columns=['Crop_Type'], drop_first=True)

# Pisahkan kembali menjadi train dan forecast data setelah encoding
train_encoded = combined_data[:len(train)]
forecast_encoded = combined_data[len(train):]

# 4. Training model Random Forest
X_train = train_encoded[['Day', 'Month', 'Year'] + [col for col in train_encoded.columns if 'Crop_Type_' in col]]  # Fitur
y_train = train_encoded['Nilai']  # Target
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Forecasting data masa depan
X_forecast = forecast_encoded[['Day', 'Month', 'Year'] + [col for col in forecast_encoded.columns if 'Crop_Type_' in col]]
forecast_encoded['Predicted'] = model.predict(X_forecast)

# 6. Gabungkan data historis dan hasil prediksi
forecast_encoded['Nilai'] = np.nan  # Data prediksi tidak memiliki nilai aktual
combined_data_final = pd.concat([data, forecast_encoded[['Tanggal', 'Nilai', 'Predicted']]], ignore_index=True)

# 7. Simpan hasil prediksi ke dalam file Excel
output_file = 'forecast_results.xlsx'
combined_data_final.to_excel(output_file, index=False)
print(f"Hasil forecast telah disimpan ke file: {output_file}")

# 8. Visualisasi hasil
plt.figure(figsize=(12, 6))
plt.plot(data['Tanggal'], data['Nilai'], label='Data Historis', marker='o')
plt.plot(forecast_df['Tanggal'], forecast_encoded['Predicted'], label='Forecast', marker='x', color='orange')
plt.axvline(x=data['Tanggal'].iloc[-1], color='red', linestyle='--', label='Mulai Forecast')
plt.title('Forecasting dengan Random Forest')
plt.xlabel('Tanggal')
plt.ylabel('Nilai')
plt.legend()
plt.grid()
plt.show()
