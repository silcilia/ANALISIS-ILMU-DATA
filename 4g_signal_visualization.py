import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned data
file_path = r'C:\Users\acer\Downloads\Kondisi_Sinyal_Telepon_Seluler_Di_Wilayah_desa_cleaned.csv'
df = pd.read_csv(file_path)

# 1. Menghitung nilai median, mean, modus untuk PERSEN 4G
mean_4g = df['PERSEN 4G'].mean()
median_4g = df['PERSEN 4G'].median()
mode_4g = df['PERSEN 4G'].mode()[0]

# Setup figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analisis Kondisi Sinyal Seluler di Wilayah Desa (Kelurahan)', fontsize=16)

# 2. Histogram untuk PERSEN 4G
axs[0, 0].hist(df['PERSEN 4G'], bins=20, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Distribusi Persen 4G')
axs[0, 0].set_xlabel('Persen 4G')
axs[0, 0].set_ylabel('Frekuensi')
axs[0, 0].grid(True)

# 3. Grafik Garis untuk PERSEN 4G
axs[0, 1].plot(df.index, df['PERSEN 4G'], color='green', marker='o', linestyle='-', markersize=4)
axs[0, 1].set_title('Grafik Garis Persen 4G')
axs[0, 1].set_xlabel('Indeks')
axs[0, 1].set_ylabel('Persen 4G')
axs[0, 1].grid(True)

# 4. Scatter Plot untuk PERSEN 4G vs LUASDESA
axs[1, 0].scatter(df['LUASDESA'], df['PERSEN 4G'], color='purple', alpha=0.6)
axs[1, 0].set_title('Scatter Plot Persen 4G vs Luas Desa')
axs[1, 0].set_xlabel('Luas Desa')
axs[1, 0].set_ylabel('Persen 4G')
axs[1, 0].grid(True)

# Menambahkan garis vertikal untuk mean, median, dan mode di scatter plot
axs[1, 0].axvline(x=mean_4g, color='r', linestyle='--', label=f'Mean: {mean_4g:.2f}')
axs[1, 0].axvline(x=median_4g, color='g', linestyle='-', label=f'Median: {median_4g:.2f}')
axs[1, 0].axvline(x=mode_4g, color='b', linestyle=':', label=f'Modus: {mode_4g:.2f}')

# Menambahkan teks di atas garis vertikal
axs[1, 0].text(mean_4g, axs[1, 0].get_ylim()[1] * 0.8, 'Mean', color='red', ha='center')
axs[1, 0].text(median_4g, axs[1, 0].get_ylim()[1] * 0.9, 'Median', color='green', ha='center')
axs[1, 0].text(mode_4g, axs[1, 0].get_ylim()[1] * 0.7, 'Modus', color='blue', ha='center')

axs[1, 0].legend()

# 5. Scatter Plot untuk PERSEN 4G vs PERSEN 5G
scatter = axs[1, 1].scatter(df['PERSEN 4G'], df['PERSEN 5G'], c=df['LUASDESA'], cmap='coolwarm', alpha=0.6)
axs[1, 1].set_title('Scatter Plot Persen 4G vs Persen 5G')
axs[1, 1].set_xlabel('Persen 4G')
axs[1, 1].set_ylabel('Persen 5G')
axs[1, 1].grid(True)

# Menambahkan colorbar
cbar = plt.colorbar(scatter, ax=axs[1, 1])
cbar.set_label('Luas Desa')

# Menyesuaikan layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Tampilkan grafik
plt.show()
