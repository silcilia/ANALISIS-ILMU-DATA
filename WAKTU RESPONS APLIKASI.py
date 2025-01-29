import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
data = {
    "time": ["08:00", "09:00", "10:00", "11:00", "12:00"],
    "active_user": [1200, 1500, 1800, 2200, 2500],
    "response_server": [1.2, 1.3, 2.1, 2.8, 3.0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculating statistics for response_server
mean_response = np.mean(df['response_server'])
median_response = np.median(df['response_server'])
std_response = np.std(df['response_server'])

# Calculating statistics for active_user
mean_active_user = np.mean(df['active_user'])
median_active_user = np.median(df['active_user'])
std_active_user = np.std(df['active_user'])

# Menampilkan hasil
print(f"Rata-rata waktu respons: {mean_response:.2f} detik")
print(f"Median waktu respons: {median_response:.2f} detik")
print(f"Standar deviasi waktu respons: {std_response:.2f} detik\n")

print(f"Rata-rata jumlah pengguna aktif: {mean_active_user:.2f}")
print(f"Median jumlah pengguna aktif: {median_active_user:.2f}")
print(f"Standar deviasi jumlah pengguna aktif: {std_active_user:.2f}")

# Create a figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for jumlah pengguna aktif
ax1.bar(df['time'], df['active_user'], color='lightblue', label='Jumlah Pengguna Aktif', alpha=0.7)
ax1.set_ylabel('Jumlah Pengguna Aktif', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xlabel('Waktu')

# Create a second y-axis for line chart of response time
ax2 = ax1.twinx()
ax2.plot(df['time'], df['response_server'], color='orange', marker='o', label='Waktu Respons', linewidth=2)
ax2.set_ylabel('Waktu Respons (detik)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Title and grid
plt.title('Pola Penggunaan Aplikasi dan Waktu Respons Server')
ax1.grid()

# Show the plot
plt.show()
