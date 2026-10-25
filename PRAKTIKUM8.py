"""
Praktikum 8: Mamdani dan Lima Metode Defuzzifikasi
Studi kasus adaptasi: penentuan harga rumah berdasarkan topik jurnal Bandung.
Catatan: jurnal sumber menggunakan Tsukamoto; praktikum ini memakai Mamdani.
Semua parameter harga dan firing strength berikut adalah asumsi simulasi.
Instalasi: pip install numpy matplotlib scikit-fuzzy
"""
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# 1. Semesta output harga rumah dalam juta rupiah
z = np.linspace(350, 550, 2001)

# 2. Fungsi keanggotaan output
mf_murah = fuzz.trapmf(z, [350, 350, 420, 480])
mf_sedang = fuzz.trimf(z, [400, 450, 500])
mf_mahal = fuzz.trapmf(z, [420, 480, 550, 550])

# 3. Simulasi firing strength
alpha_murah = 0.5
alpha_sedang = 0.8
alpha_mahal = 0.2

# 4. Implikasi Mamdani: MIN (clipping)
imp_murah = np.fmin(alpha_murah, mf_murah)
imp_sedang = np.fmin(alpha_sedang, mf_sedang)
imp_mahal = np.fmin(alpha_mahal, mf_mahal)

# 5. Agregasi: MAX
agregasi = np.fmax(imp_murah, np.fmax(imp_sedang, imp_mahal))

# 6. Lima metode defuzzifikasi menggunakan scikit-fuzzy
metode = {
    "Centroid": "centroid",
    "Bisector": "bisector",
    "MOM": "mom",
    "SOM": "som",
    "LOM": "lom",
}
hasil = {}
for nama, kode in metode.items():
    hasil[nama] = fuzz.defuzz(z, agregasi, kode)

print("\nHASIL DEFUZZIFIKASI MAMDANI — HARGA RUMAH")
print("-" * 58)
print(f"{'Metode':<15}{'Juta rupiah':>18}{'Perkiraan rupiah':>24}")
for nama, nilai in hasil.items():
    print(f"{nama:<15}{nilai:>18.2f}{('Rp' + format(round(nilai*1_000_000), ',').replace(',', '.')):>24}")

# 7. Grafik visualisasi
plt.figure(figsize=(11, 6.5))
plt.plot(z, mf_murah, label="Murah (fungsi asal)")
plt.plot(z, mf_sedang, label="Sedang (fungsi asal)")
plt.plot(z, mf_mahal, label="Mahal (fungsi asal)")
plt.plot(z, agregasi, linewidth=2.5, label="Agregasi Mamdani (MIN-MAX)")
for nama, nilai in hasil.items():
    plt.axvline(nilai, linestyle="--", linewidth=1.5,
                label=f"{nama} = {nilai:.2f}")
plt.title("Perbandingan Lima Metode Defuzzifikasi — Harga Rumah")
plt.xlabel("Harga rumah (juta rupiah)")
plt.ylabel("Derajat keanggotaan")
plt.xlim(350, 550)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("Grafik_Mamdani_Harga_Rumah.png", dpi=300)
plt.show()
