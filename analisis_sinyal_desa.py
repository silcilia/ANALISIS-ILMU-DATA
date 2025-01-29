import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned data
file_path = r'C:\Users\acer\Downloads\Kondisi_Sinyal_Telepon_Seluler_Di_Wilayah_desa_cleaned.csv'
df = pd.read_csv(file_path)

### a. Business Understanding ###
# Langkah ini melibatkan identifikasi tujuan bisnis, kebutuhan bisnis, dan proses bisnis yang sudah ada.
# Tidak diperlukan kode karena ini adalah proses analisis bisnis secara manual.

### b. Menelaah Data ###

# 1. Analisis Tipe dan Relasi Data
print("Tipe Data:")
print(df.dtypes)

print("\nKorelasi antar variabel:")
correlation_matrix = df[['LUASDESA', 'LUAS_MUKIM', 'PERSEN 2G', 'PERSEN 3G', 'PERSEN 4G', 'PERSEN 5G']].corr()
print(correlation_matrix)

# 2. Analisis Karakteristik Data
print("\nStatistik Deskriptif:")
print(df.describe())

# Visualisasi distribusi sinyal
df[['PERSEN 2G', 'PERSEN 3G', 'PERSEN 4G', 'PERSEN 5G']].hist(bins=20, figsize=(10, 8), layout=(2, 2))
plt.suptitle("Distribusi Persentase Sinyal (2G, 3G, 4G, 5G)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

### c. Memvalidasi Data ###

# Mengecek jumlah nilai NaN setelah pembersihan
nan_count = df.isna().sum().sum()
print(f"\nJumlah nilai NaN setelah pembersihan: {nan_count}")

# Jumlah missing values per kolom
missing_values = df.isna().sum()
print("\nJumlah missing values per kolom:")
print(missing_values)

### d. Pembersihan Data ###

# Jika masih ada nilai yang hilang, lakukan pembersihan
df_cleaned = df.dropna()  # Menghapus baris dengan missing values
# Atau bisa menggunakan metode pengisian:
# df_filled = df.fillna(df.mean())  # Mengisi missing values dengan rata-rata

### e. Visualisasi Data (Mean, Median, Modus) dan Interpretasinya ###

# 1. Menghitung Mean, Median, Modus untuk PERSEN 4G
mean_4g = df['PERSEN 4G'].mean()
median_4g = df['PERSEN 4G'].median()
mode_4g = df['PERSEN 4G'].mode()[0]

print(f"\nMean 4G: {mean_4g}")
print(f"Median 4G: {median_4g}")
print(f"Mode 4G: {mode_4g}")

# 2. Visualisasi Distribusi dan Scatter Plot
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analisis Kondisi Sinyal Seluler di Wilayah Desa (Kelurahan)', fontsize=16)

# Histogram untuk PERSEN 4G
axs[0, 0].hist(df['PERSEN 4G'], bins=20, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Distribusi Persen 4G')
axs[0, 0].set_xlabel('Persen 4G')
axs[0, 0].set_ylabel('Frekuensi')
axs[0, 0].grid(True)

# Grafik Garis untuk PERSEN 4G
axs[0, 1].plot(df.index, df['PERSEN 4G'], color='green', marker='o', linestyle='-', markersize=4)
axs[0, 1].set_title('Grafik Garis Persen 4G')
axs[0, 1].set_xlabel('Indeks')
axs[0, 1].set_ylabel('Persen 4G')
axs[0, 1].grid(True)

# Scatter Plot untuk PERSEN 4G vs LUASDESA
axs[1, 0].scatter(df['LUASDESA'], df['PERSEN 4G'], color='purple', alpha=0.6)
axs[1, 0].set_title('Scatter Plot Persen 4G vs Luas Desa')
axs[1, 0].set_xlabel('Luas Desa')
axs[1, 0].set_ylabel('Persen 4G')
axs[1, 0].grid(True)

# Menambahkan garis vertikal untuk Mean, Median, dan Modus di scatter plot
axs[1, 0].axvline(x=mean_4g, color='r', linestyle='--', label=f'Mean: {mean_4g:.2f}')
axs[1, 0].axvline(x=median_4g, color='g', linestyle='-', label=f'Median: {median_4g:.2f}')
axs[1, 0].axvline(x=mode_4g, color='b', linestyle=':', label=f'Modus: {mode_4g:.2f}')

axs[1, 0].legend()

# Scatter Plot untuk PERSEN 4G vs PERSEN 5G
scatter = axs[1, 1].scatter(df['PERSEN 4G'], df['PERSEN 5G'], c=df['LUASDESA'], cmap='coolwarm', alpha=0.6)
axs[1, 1].set_title('Scatter Plot Persen 4G vs Persen 5G')
axs[1, 1].set_xlabel('Persen 4G')
axs[1, 1].set_ylabel('Persen 5G')
axs[1, 1].grid(True)

# Menambahkan colorbar untuk menunjukkan Luas Desa
cbar = plt.colorbar(scatter, ax=axs[1, 1])
cbar.set_label('Luas Desa')

# Menyesuaikan layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
