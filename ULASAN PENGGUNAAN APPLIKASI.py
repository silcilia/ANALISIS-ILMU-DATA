import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Dataset ulasan pengguna
data = {
    "No": [1, 2, 3, 4, 5],
    "Ulasan Pengguna": [
        "Aplikasi ini bagus, tetapi sering lambat saat sibuk.",
        "Proses checkout terlalu lambat. Perlu perbaikan.",
        "Saya suka tampilannya, tetapi fitur pencarian agak sulit digunakan.",
        "Layanan pelanggan baik, tetapi aplikasi sering crash.",
        "Cepat dan mudah digunakan, sangat membantu."
    ]
}

# Buat DataFrame
df = pd.DataFrame(data)

# Kategorikan ulasan dan berikan kode
def categorize_review(review):
    if "lambat" in review or "crash" in review:
        return "Keluhan Performa", "Lambat"  # Kode: Lambat
    elif "tampilan" in review or "fitur" in review:
        return "Desain Antarmuka", "Tampilan"  # Kode: Tampilan
    elif "mudah" in review or "baik" in review:
        return "Pengalaman Pengguna", "Mudah"  # Kode: Mudah
    else:
        return "Lainnya", "Lainnya"  # Kode: Lainnya

# Terapkan fungsi kategori
df[['Tema', 'Kode']] = df['Ulasan Pengguna'].apply(lambda x: pd.Series(categorize_review(x)))

# Menampilkan DataFrame dengan kategori dan kode
print(df)

# Mengidentifikasi pola yang muncul
pola = df['Tema'].value_counts()
print("\nPola yang muncul:")
print(pola)

# Dataset Ulasan Pengguna untuk Word Cloud
ulasan = df['Ulasan Pengguna'].tolist()

# Gabungkan semua ulasan menjadi satu string
text = " ".join(ulasan)

# Buat word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                      colormap='viridis', max_words=100).generate(text)

# Tampilkan word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Menyembunyikan sumbu
plt.title('Word Cloud dari Ulasan Pengguna', fontsize=16)
plt.show()
