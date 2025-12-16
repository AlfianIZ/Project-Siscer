import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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

diabetes_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
diabetes_sim = ctrl.ControlSystemSimulation(diabetes_ctrl)


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
        
        if skor <= 45: kat = "Rendah"
        elif skor <= 65: kat = "Sedang"
        else: kat = "Tinggi"
        
        return skor, kat 
    except:
        return 0, "Error"

def tampilkan_grafik_batang(df_hasil, acc):

    kategori_order = ['Rendah', 'Sedang', 'Tinggi']
    
    fuzzy_counts = df_hasil['Fuzzy_Prediksi'].value_counts().reindex(kategori_order, fill_value=0)
    
    label_counts = df_hasil['Label_Baru'].value_counts().reindex(kategori_order, fill_value=0)

    x = np.arange(len(kategori_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, label_counts, width, label='Label Asli (Dataset)', color='gray', alpha=0.7)
    rects2 = ax.bar(x + width/2, fuzzy_counts, width, label='Prediksi Fuzzy', color=['green', 'orange', 'red'])

    ax.set_ylabel('Jumlah Pasien')
    ax.set_title(f'Distribusi Hasil Diagnosa (Akurasi: {acc*100:.2f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(kategori_order)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


def tampilkan_grafik_keanggotaan():
    def plot_window(variable, title, xlabel):
        plt.figure(figsize=(8, 5))
        for label in variable.terms:
            plt.plot(variable.universe, variable[label].mf, label=label, linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold', color='darkblue')
        plt.xlabel(xlabel)
        plt.ylabel('Derajat Keanggotaan (µ)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plot_window(Diabetes, 'Fungsi Keanggotaan Output Risiko', 'Skor Risiko (0-100)')
    plot_window(kadar_gula_darah, 'Fungsi Keanggotaan Gula Darah', 'Gula Darah (mg/dL)')
    plot_window(umur, 'Fungsi Keanggotaan Umur', 'Umur (Tahun)')
    plot_window(BMI, 'Fungsi Keanggotaan BMI', 'Nilai BMI')

    plt.show()

def main():
    nama_file_bersih = "diabetes_cleaned.csv"
    
    try:
        print("Memuat data...")
        df = pd.read_csv(nama_file_bersih)
        print(f"Berhasil! Memproses SELURUH DATA ({len(df)} baris)...")

        df_run = df.copy()
        hasil = []
        for idx, row in df_run.iterrows():
            if idx % 5000 == 0 and idx > 0:
                print(f"Sedang memproses baris ke-{idx}...")

            _, kat = prediksi_fuzzy(row['bmi'], row['age'], row['blood_glucose_level'])
            hasil.append(kat)
            
        df_run['Fuzzy_Prediksi'] = hasil
        
        # Hitung Akurasi
        acc = accuracy_score(df_run['Label_Baru'], df_run['Fuzzy_Prediksi'])
        print("\n" + "="*40)
        print(f" AKURASI SISTEM (TOTAL): {acc*100:.2f}%")
        print("="*40)
        
        # GRAFIK BATANG 
        print("Menampilkan Grafik Distribusi Akurasi...")
        tampilkan_grafik_batang(df_run, acc)
        
        # GRAFIK KEANGGOTAAN
        print("\nMenampilkan Grafik Keanggotaan Fuzzy...")
        tampilkan_grafik_keanggotaan()
        
    except FileNotFoundError:
        print(f"ERROR: File '{nama_file_bersih}' tidak ditemukan.")
        print("Harap jalankan kode 'cleaning_data.py' terlebih dahulu!")

if __name__ == "__main__":
    main()
