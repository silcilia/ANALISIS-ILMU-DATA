import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Membaca data dan mengubah data yang hilang
file_path = r'C:\Users\acer\Downloads\Kondisi_Sinyal_Telepon_Seluler_Di_Wilayah_desa_cleaned.csv'
data = pd.read_csv(file_path)
data.replace("TIDAK TERDAPAT INFORMASI AREA PEMUKIMAN", None, inplace=True)

# Ubah kolom yang relevan ke tipe numerik
data['PERSEN 4G'] = pd.to_numeric(data['PERSEN 4G'], errors='coerce')
data['LUASDESA'] = pd.to_numeric(data['LUASDESA'], errors='coerce')
data['LUAS_MUKIM'] = pd.to_numeric(data['LUAS_MUKIM'], errors='coerce')

# Drop rows dengan missing values pada kolom yang dipilih
data = data.dropna(subset=['PERSEN 4G', 'LUASDESA', 'LUAS_MUKIM'])

# Pilih variabel independen dan dependen
X = data[['LUASDESA', 'LUAS_MUKIM']]
y = data['PERSEN 4G']

# Bagi data ke dalam train dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bangun model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
