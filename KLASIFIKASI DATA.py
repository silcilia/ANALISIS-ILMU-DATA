import pandas as pd
from io import StringIO

# Data dalam format string
data_string = """
Farm_ID,Crop_Type,Farm_Area(acres),Irrigation_Type,Fertilizer_Used(tons),Pesticide_Used(kg),Yield(tons),Soil_Type,Season,Water_Usage(cubic meters)
F001,Cotton,329.4,Sprinkler,8.14,2.21,14.44,Loamy,Kharif,76648.2
F002,Carrot,18.67,Manual,4.77,4.36,42.91,Peaty,Kharif,68725.54
F003,Sugarcane,306.03,Flood,2.91,0.56,33.44,Silty,Kharif,75538.56
F004,Tomato,380.21,Rain-fed,3.32,4.35,34.08,Silty,Zaid,45401.23
F005,Tomato,135.56,Sprinkler,8.33,4.48,43.28,Clay,Zaid,93718.69
F006,Sugarcane,12.5,Sprinkler,6.42,2.25,38.18,Loamy,Zaid,46487.98
F007,Soybean,360.06,Drip,1.83,2.37,44.93,Sandy,Rabi,40583.57
F008,Rice,464.6,Drip,5.18,0.91,4.23,Silty,Kharif,9392.38
F009,Maize,389.37,Drip,0.57,4.93,3.86,Peaty,Rabi,60202.14
F010,Soybean,184.37,Drip,2.18,2.67,17.25,Sandy,Kharif,90922.15
"""

# Memuat data menggunakan StringIO
data = pd.read_csv(StringIO(data_string))

# Menambahkan kolom klasifikasi berdasarkan Yield(tons)
data['Yield_Category'] = data['Yield(tons)'].apply(lambda x: 'High' if x > 30 else 'Low')

# Mengatur opsi untuk menampilkan seluruh baris
pd.set_option('display.max_rows', None)

# Menampilkan seluruh dataset
print(data[['Farm_ID', 'Yield(tons)', 'Yield_Category']])

# Menyimpan dataset yang telah diklasifikasikan ke file baru
data.to_csv("classified_dataset.csv", index=False)

print("Klasifikasi selesai. File hasil klasifikasi disimpan sebagai 'classified_dataset.csv'")
