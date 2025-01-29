import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Memuat dataset
file_path = r'C:\Users\acer\Downloads\Kondisi_Sinyal_Telepon_Seluler_Di_Wilayah_desa_cleaned.csv'
data = pd.read_csv(file_path)

# Menampilkan informasi dasar tentang dataset
print("Informasi dataset:")
print(data.info())
print("\nBeberapa baris pertama dari dataset:")
print(data.head())

# Pra-pemrosesan: Mengonversi kolom persentase ke nilai numerik
columns_to_convert = ['PERSEN 2G', 'PERSEN 3G', 'PERSEN 4G', 'PERSEN 5G', 
                      'PERSEN MUK 2G', 'PERSEN MUK 3G', 'PERSEN MUK 4G', 'PERSEN MUK 5G',
                      'PER 2G', 'PER 3G', 'PER 4G', 'PER 5G', 
                      'PER MUK 2G', 'PER MUK 3G', 'PER MUK 4G', 'PER MUK 5G']

# Mengonversi kolom persentase ke numerik
for col in columns_to_convert:
    data[col] = data[col].astype(str).str.replace('%', '').str.replace(',', '.')
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Menghapus baris dengan nilai NaN
data.dropna(subset=columns_to_convert, inplace=True)
print("\nJumlah data setelah menghapus NaN:", data.shape)

# Menentukan variabel independen (X) dan dependen (y)
X = data.drop(columns=['PERSEN 4G'])
y = data['PERSEN 4G']

# Memeriksa dan menangani data non-numerik di fitur
X = X.apply(pd.to_numeric, errors='coerce')
X.dropna(inplace=True)

# Memastikan bahwa variabel target (y) disesuaikan dengan fitur (X)
y = y[X.index]  # pastikan y sesuai dengan X

# Memastikan ada data sebelum pemisahan
print("\nJumlah fitur X sebelum pemisahan:", X.shape)
print("Jumlah target y sebelum pemisahan:", y.shape)

# Memeriksa apakah ada data yang tersisa
if X.empty or y.empty:
    print("Data X atau y kosong. Periksa proses pembersihan data.")
else:
    # Memisahkan data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Random Forest Regressor
    print("\nMelatih model Random Forest...")
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Melakukan prediksi dengan Random Forest
    rf_predictions = rf_model.predict(X_test)

    # Menghitung dan menampilkan error metrics untuk Random Forest
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_mse = mean_squared_error(y_test, rf_predictions)

    print("\nRandom Forest Regressor:")
    print("Mean Absolute Error:", rf_mae)
    print("Mean Squared Error:", rf_mse)

    # Model 2: Linear Regression
    print("\nMelatih model Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Melakukan prediksi dengan Linear Regression
    lr_predictions = lr_model.predict(X_test)

    # Menghitung dan menampilkan error metrics untuk Linear Regression
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    lr_mse = mean_squared_error(y_test, lr_predictions)

    print("\nLinear Regression:")
    print("Mean Absolute Error:", lr_mae)
    print("Mean Squared Error:", lr_mse)

    # Menampilkan hasil prediksi
    predicted_df = pd.DataFrame({'Actual': y_test, 
                                  'Random Forest Predicted': rf_predictions, 
                                  'Linear Regression Predicted': lr_predictions})
    print("\nHasil Prediksi:")
    print(predicted_df.head())  # Tampilkan 5 baris pertama dari hasil prediksi

    # Contoh prediksi masa depan dengan Random Forest
    new_data = {
        'PERSEN 2G': [50.0], 
        'PERSEN 3G': [40.0], 
        'PERSEN 5G': [10.0], 
        'PERSEN MUK 2G': [30.0], 
        'PERSEN MUK 3G': [25.0], 
        'PERSEN MUK 4G': [80.0], 
        'PERSEN MUK 5G': [10.0],
        'PER 2G': [75.0], 
        'PER 3G': [65.0], 
        'PER 4G': [80.0], 
        'PER 5G': [30.0],
        'PER MUK 2G': [70.0], 
        'PER MUK 3G': [60.0], 
        'PER MUK 4G': [90.0], 
        'PER MUK 5G': [20.0]
    }

    new_df = pd.DataFrame(new_data)

    # Prediksi masa depan untuk PERSEN 4G berdasarkan data baru
    future_prediction = rf_model.predict(new_df)
    print("Prediksi Masa Depan (PERSEN 4G):", future_prediction[0])
