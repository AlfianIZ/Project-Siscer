import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ============================================================
# 1. KONFIGURASI ENGINE FUZZY
# ============================================================

# Universe
BMI = ctrl.Antecedent(np.arange(0, 101, 1), 'BMI')
umur = ctrl.Antecedent(np.arange(0, 121, 1), 'Umur')
kadar_gula_darah = ctrl.Antecedent(np.arange(0, 601, 1), 'Gula Darah')
Diabetes = ctrl.Consequent(np.arange(0, 101, 1), 'Risiko Diabetes')

# Membership Functions
BMI['Rendah'] = fuzz.trapmf(BMI.universe, [0, 0, 18.5, 25])
BMI['Sedang'] = fuzz.trimf(BMI.universe, [23, 27, 32])
BMI['Tinggi'] = fuzz.trapmf(BMI.universe, [28, 35, 100, 100])

umur['Rendah'] = fuzz.trapmf(umur.universe, [0, 0, 30, 45])
umur['Sedang'] = fuzz.trimf(umur.universe, [35, 50, 65])
umur['Tinggi'] = fuzz.trapmf(umur.universe, [55, 70, 120, 120])

kadar_gula_darah['Rendah'] = fuzz.trapmf(kadar_gula_darah.universe, [0, 0, 100, 130])
kadar_gula_darah['Sedang'] = fuzz.trimf(kadar_gula_darah.universe, [110, 150, 190])
kadar_gula_darah['Tinggi'] = fuzz.trapmf(kadar_gula_darah.universe, [160, 200, 600, 600])

Diabetes['Rendah'] = fuzz.trimf(Diabetes.universe, [0, 0, 50])
Diabetes['Sedang'] = fuzz.trimf(Diabetes.universe, [30, 50, 70])
Diabetes['Tinggi'] = fuzz.trimf(Diabetes.universe, [50, 100, 100])

# Rules
rule1 = ctrl.Rule(kadar_gula_darah['Tinggi'], Diabetes['Tinggi'])
rule2 = ctrl.Rule(kadar_gula_darah['Sedang'] & (BMI['Tinggi'] | umur['Tinggi']), Diabetes['Tinggi'])
rule3 = ctrl.Rule(kadar_gula_darah['Sedang'] & (BMI['Sedang'] | BMI['Rendah']), Diabetes['Sedang'])
rule4 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Tinggi'] & umur['Tinggi'], Diabetes['Sedang'])
rule5 = ctrl.Rule(kadar_gula_darah['Rendah'] & (BMI['Rendah'] | BMI['Sedang']), Diabetes['Rendah'])
rule6 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Tinggi'] & (umur['Rendah'] | umur['Sedang']), Diabetes['Rendah'])

# Build System
diabetes_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
diabetes_sim = ctrl.ControlSystemSimulation(diabetes_ctrl)

# ============================================================
# 2. FUNGSI PREDIKSI
# ============================================================

def prediksi_fuzzy(bmi, umur, gula):
    try:
        b = np.clip(float(bmi), 0, 100)
        u = np.clip(float(umur), 0, 120)
        g = np.clip(float(gula), 0, 600)
        
        diabetes_sim.input['BMI'] = b
        diabetes_sim.input['Umur'] = u
        diabetes_sim.input['Gula Darah'] = g
        
        diabetes_sim.compute()
        skor = diabetes_sim.output['Risiko Diabetes']
        
        if skor <= 45: return "Rendah"
        elif skor <= 65: return "Sedang"
        else: return "Tinggi"
    except:
        return "Error"

# ============================================================
# 3. FUNGSI GRAFIK BATANG (BAR CHART)
# ============================================================

def tampilkan_grafik_batang(df_hasil, acc):
    # Hitung jumlah per kategori
    kategori_order = ['Rendah', 'Sedang', 'Tinggi']
    
    # Hitung hasil prediksi Fuzzy
    fuzzy_counts = df_hasil['Fuzzy_Prediksi'].value_counts().reindex(kategori_order, fill_value=0)
    
    # Hitung label asli (Dataset)
    label_counts = df_hasil['Label_Baru'].value_counts().reindex(kategori_order, fill_value=0)

    # Setup Plot
    x = np.arange(len(kategori_order))
    width = 0.35  # Lebar batang

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dua batang berdampingan
    rects1 = ax.bar(x - width/2, label_counts, width, label='Label Asli (Dataset)', color='gray', alpha=0.7)
    rects2 = ax.bar(x + width/2, fuzzy_counts, width, label='Prediksi Fuzzy', color=['green', 'orange', 'red'])

    # Label dan Judul
    ax.set_ylabel('Jumlah Pasien')
    ax.set_title(f'Distribusi Hasil Diagnosa (Akurasi: {acc*100:.2f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(kategori_order)
    ax.legend()

    # Fungsi untuk menampilkan angka di atas batang
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()

# ============================================================
# 4. MAIN PROGRAM
# ============================================================

def main():
    nama_file_bersih = "diabetes_cleaned.csv"
    
    try:
        print("Mencoba memuat data bersih...")
        df = pd.read_csv(nama_file_bersih)
        
        # --- PERUBAHAN DI SINI ---
        # Kita pakai df.copy() untuk mengambil SEMUA data
        print(f"Berhasil! Memproses SELURUH DATA ({len(df)} baris) dengan Fuzzy Logic...")
        print("Mohon tunggu, proses ini mungkin memakan waktu agak lama...")
        
        df_run = df.copy() 
        # -------------------------
        
        hasil = []
        # Menggunakan enumerate agar bisa print progress setiap 5000 data
        for idx, row in df_run.iterrows():
            if idx % 5000 == 0 and idx > 0:
                print(f"Sedang memproses baris ke-{idx}...")
                
            res = prediksi_fuzzy(row['bmi'], row['age'], row['blood_glucose_level'])
            hasil.append(res)
            
        df_run['Fuzzy_Prediksi'] = hasil
        print("Proses Fuzzy selesai!")
        
        # Hitung Akurasi
        acc = accuracy_score(df_run['Label_Baru'], df_run['Fuzzy_Prediksi'])
        print("\n" + "="*40)
        print(f" AKURASI SISTEM (TOTAL): {acc*100:.2f}%")
        print("="*40)
        
        # TAMPILKAN GRAFIK BATANG
        print("Menampilkan grafik distribusi...")
        tampilkan_grafik_batang(df_run, acc)
        
        # Input Manual
        print("\n--- MODE INPUT MANUAL ---")
        while True:
            tanya = input("Tes diagnosa pasien? (y/n): ").lower()
            if tanya != 'y': break
            try:
                b = float(input("BMI : "))
                u = float(input("Umur: "))
                g = float(input("Gula: "))
                print(f">>> HASIL: {prediksi_fuzzy(b, u, g)}\n")
            except:
                print("Input harus angka.")

    except FileNotFoundError:
        print(f"ERROR: File '{nama_file_bersih}' tidak ditemukan.")
        print("Harap jalankan kode 'cleaning_data.py' terlebih dahulu!")

if __name__ == "__main__":
    main()