import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI FUZZY LOGIC (ENGINE)
# ==========================================

# --- A. Universe (Rentang Data) ---
BMI = ctrl.Antecedent(np.arange(0, 101, 1), 'BMI')
umur = ctrl.Antecedent(np.arange(0, 121, 1), 'Umur')
kadar_gula_darah = ctrl.Antecedent(np.arange(0, 601, 1), 'Gula Darah')
Diabetes = ctrl.Consequent(np.arange(0, 101, 1), 'Risiko Diabetes')

# --- B. Membership Functions ---
# BMI
BMI['Rendah'] = fuzz.trapmf(BMI.universe, [0, 0, 18.5, 24])
BMI['Sedang'] = fuzz.trimf(BMI.universe, [22, 27, 32])
BMI['Tinggi'] = fuzz.trapmf(BMI.universe, [28, 35, 100, 100])

# Umur
umur['Rendah'] = fuzz.trapmf(umur.universe, [0, 0, 30, 45])
umur['Sedang'] = fuzz.trimf(umur.universe, [35, 50, 65])
umur['Tinggi'] = fuzz.trapmf(umur.universe, [55, 70, 120, 120])

# Gula Darah
kadar_gula_darah['Rendah'] = fuzz.trapmf(kadar_gula_darah.universe, [0, 0, 100, 145])
kadar_gula_darah['Sedang'] = fuzz.trimf(kadar_gula_darah.universe, [130, 165, 200])
kadar_gula_darah['Tinggi'] = fuzz.trapmf(kadar_gula_darah.universe, [180, 220, 600, 600])

# Output Risiko
Diabetes['Rendah'] = fuzz.trimf(Diabetes.universe, [0, 0, 50])
Diabetes['Sedang'] = fuzz.trimf(Diabetes.universe, [25, 50, 75])
Diabetes['Tinggi'] = fuzz.trimf(Diabetes.universe, [50, 100, 100])

# --- C. Rules ---
rule1 = ctrl.Rule(kadar_gula_darah['Tinggi'], Diabetes['Tinggi'])
rule2 = ctrl.Rule(kadar_gula_darah['Sedang'] & (BMI['Tinggi'] | umur['Tinggi']), Diabetes['Tinggi'])
rule3 = ctrl.Rule(kadar_gula_darah['Sedang'] & (BMI['Sedang'] | BMI['Rendah']), Diabetes['Sedang'])
rule4 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Tinggi'] & umur['Tinggi'], Diabetes['Sedang'])
rule5 = ctrl.Rule(kadar_gula_darah['Rendah'] & (BMI['Rendah'] | BMI['Sedang']), Diabetes['Rendah'])
rule6 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Tinggi'] & (umur['Rendah'] | umur['Sedang']), Diabetes['Rendah'])

diabetes_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
diabetes_sim = ctrl.ControlSystemSimulation(diabetes_ctrl)

# ==========================================
# 2. FUNGSI UTAMA (PREDIKSI & GRAFIK)
# ==========================================

def kategorisasi_risiko(skor):
    if skor <= 45: return "Rendah"
    elif skor <= 75: return "Sedang"
    else: return "Tinggi"

def prediksi_aman(input_bmi, input_umur, input_gula):
    try:
        # Clipping data (Safety)
        b = np.clip(float(input_bmi), 0, 100)
        u = np.clip(float(input_umur), 0, 120)
        g = np.clip(float(input_gula), 0, 600)
        
        diabetes_sim.input['BMI'] = b
        diabetes_sim.input['Umur'] = u
        diabetes_sim.input['Gula Darah'] = g
        
        diabetes_sim.compute()
        hasil_skor = diabetes_sim.output['Risiko Diabetes']
        return hasil_skor, kategorisasi_risiko(hasil_skor)
    except Exception as e:
        return -1.0, "Error"

def tampilkan_grafik_detail(input_bmi, input_umur, input_gula, output_skor, output_label):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Analisa Individu: {output_label} (Skor: {output_skor:.2f})', fontsize=16, fontweight='bold', color='blue')

    def plot_mf(ax, var, val, title):
        for label in var.terms:
            ax.plot(var.universe, var[label].mf, label=label, linewidth=2)
        if val is not None:
            ax.axvline(x=val, color='k', linestyle='--', linewidth=2, label='Input')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_mf(ax1, BMI, input_bmi, 'BMI')
    plot_mf(ax2, umur, input_umur, 'Umur')
    plot_mf(ax3, kadar_gula_darah, input_gula, 'Gula Darah')
    
    # Plot Output
    for label in Diabetes.terms:
        ax4.plot(Diabetes.universe, Diabetes[label].mf, label=label, linewidth=2)
    ax4.axvline(x=output_skor, color='red', linewidth=3, label='Hasil')
    ax4.fill_between(Diabetes.universe, 0, 1, where=(Diabetes.universe >= output_skor-1) & (Diabetes.universe <= output_skor+1), color='red', alpha=0.3)
    ax4.set_title('Risiko Diabetes')
    ax4.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# 3. BAGIAN 1: PROSES DATA CSV & HITUNG AKURASI
# ==========================================

def proses_batch_csv():
    print("\n" + "="*50)
    print(" BAGIAN 1: PEMROSESAN DATA & AKURASI ")
    print("="*50)
    
    file_path = "diabetes_cleaned_with_label.csv"
    try:
        df = pd.read_csv(file_path)
        # Bersihkan data
        for col in ['bmi', 'age', 'blood_glucose_level']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = df.dropna(subset=['bmi', 'age', 'blood_glucose_level']).copy()
        print(f"Data berhasil dimuat: {len(df_clean)} baris.")
        
        # --- PERSIAPAN AKURASI ---
        # Kita perlu kamus untuk menerjemahkan label Inggris di CSV ke Indonesia
        # CSV: Low, Medium, High -> Fuzzy: Rendah, Sedang, Tinggi
        map_label = {'Low': 'Rendah', 'Medium': 'Sedang', 'High': 'Tinggi'}
        
        # Cek apakah kolom target 'risk_label' ada
        ada_label_asli = 'risk_label' in df_clean.columns
        if ada_label_asli:
            print("Kolom 'risk_label' ditemukan. Menghitung akurasi...")
            df_clean['Label_Asli_Indo'] = df_clean['risk_label'].map(map_label)
        else:
            print("Kolom 'risk_label' TIDAK ditemukan. Akurasi tidak bisa dihitung.")

        hasil_skor = []
        hasil_kat = []
        
        print("Sedang memproses Fuzzy Logic...")
        for idx, row in df_clean.iterrows():
            s, k = prediksi_aman(row['bmi'], row['age'], row['blood_glucose_level'])
            hasil_skor.append(s)
            hasil_kat.append(k)
            
        df_clean['Fuzzy_Score'] = hasil_skor
        df_clean['Fuzzy_Risk'] = hasil_kat
        
        # --- HITUNG AKURASI ---
        if ada_label_asli:
            # Bandingkan hasil prediksi dengan label asli
            df_clean['Match'] = df_clean['Fuzzy_Risk'] == df_clean['Label_Asli_Indo']
            
            jumlah_benar = df_clean['Match'].sum()
            total_data = len(df_clean)
            akurasi = (jumlah_benar / total_data) * 100
            
            print("\n" + "-"*30)
            print(f" LAPORAN AKURASI ")
            print("-" * 30)
            print(f"Total Data   : {total_data}")
            print(f"Prediksi Benar: {jumlah_benar}")
            print(f"AKURASI      : {akurasi:.2f}%")
            print("-" * 30)
            
            # Tampilkan detail salah prediksi (opsional)
            salah = df_clean[~df_clean['Match']]
            if len(salah) > 0:
                print(f"\nContoh {min(5, len(salah))} Data yang Salah Prediksi:")
                print(salah[['bmi', 'age', 'blood_glucose_level', 'Label_Asli_Indo', 'Fuzzy_Risk']].head())
        
        # Tampilkan Grafik Distribusi
        print("\nMenampilkan grafik distribusi hasil prediksi...")
        plt.figure(figsize=(10, 5))
        
        # Plot 1: Hasil Fuzzy
        plt.subplot(1, 2, 1)
        counts = df_clean[df_clean['Fuzzy_Score'] != -1.0]['Fuzzy_Risk'].value_counts()
        colors = {'Rendah':'green', 'Sedang':'orange', 'Tinggi':'red'}
        plt.bar(counts.index, counts.values, color=[colors.get(x, 'gray') for x in counts.index])
        plt.title('Hasil Prediksi Fuzzy')
        plt.xlabel('Kategori')
        plt.ylabel('Jumlah')
        
        # Plot 2: Label Asli (Jika ada)
        if ada_label_asli:
            plt.subplot(1, 2, 2)
            counts_real = df_clean['Label_Asli_Indo'].value_counts()
            plt.bar(counts_real.index, counts_real.values, color=[colors.get(x, 'gray') for x in counts_real.index])
            plt.title('Label Asli (Dataset)')
            plt.xlabel('Kategori')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("PERINGATAN: File CSV tidak ditemukan. Melewati proses batch.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# ==========================================
# 4. BAGIAN 2: INPUT MANUAL (INTERAKTIF)
# ==========================================

def proses_interaktif():
    print("\n" + "="*50)
    print(" BAGIAN 2: DIAGNOSA MANUAL (INPUT USER) ")
    print("="*50)
    
    while True:
        try:
            print("\nMasukkan Data Pasien Baru:")
            in_bmi = float(input("1. Masukkan BMI (contoh: 24.5) : "))
            in_umur = float(input("2. Masukkan Umur (tahun)       : "))
            in_gula = float(input("3. Masukkan Gula Darah (mg/dL) : "))
            
            skor, kategori = prediksi_aman(in_bmi, in_umur, in_gula)
            
            print("\n" + "-"*30)
            print(f"HASIL: Skor {skor:.2f} / 100 ({kategori})")
            print("-"*30)
            
            print("Menampilkan grafik detail keanggotaan...")
            tampilkan_grafik_detail(in_bmi, in_umur, in_gula, skor, kategori)
            
            lagi = input("\nDiagnosa pasien lain? (y/n): ").lower()
            if lagi != 'y':
                break
        except ValueError:
            print("Input tidak valid. Harap masukkan angka.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    proses_batch_csv()
    
    tanya = input("\nApakah Anda ingin lanjut ke Input Manual? (y/n): ").lower()
    if tanya == 'y':
        proses_interaktif()
    
    print("\nProgram Selesai.")