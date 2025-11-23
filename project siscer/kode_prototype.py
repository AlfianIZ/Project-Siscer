import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

file_path = "diabetes_cleaned_with_label.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("File CSV tidak ditemukan, pastikan nama file benar.")
    # Dummy data jika file tidak ada, agar program tidak crash saat setup
    df = pd.DataFrame({'bmi': [20, 30], 'age': [20, 60], 'blood_glucose_level': [80, 200]})

# Range Universe
BMI_min = int(df['bmi'].min()-5)
BMI_max = int(df['bmi'].max()+5)
umur_min = max(0, int(df['age'].min()-5))
umur_max = int(df['age'].max()+5)
kadar_gula_darah_min = int(df['blood_glucose_level'].min()-5)
kadar_gula_darah_max = int(df['blood_glucose_level'].max()+5)

# Variabel Antecedent (Input) & Consequent (Output)
BMI = ctrl.Antecedent(np.arange(BMI_min, BMI_max + 1, 1), 'BMI')
umur = ctrl.Antecedent(np.arange(umur_min, umur_max + 1, 1), 'umur')
kadar_gula_darah = ctrl.Antecedent(np.arange(kadar_gula_darah_min, kadar_gula_darah_max + 1, 1), 'kadar_gula_darah')
Diabetes = ctrl.Consequent(np.arange(0, 101, 1), 'pred_diabetes')

# Membership Functions
BMI.automf(3, names=['Rendah', 'Sedang', 'Tinggi'])
umur.automf(3, names=['Rendah', 'Sedang', 'Tinggi'])
kadar_gula_darah.automf(3, names=['Rendah', 'Sedang', 'Tinggi'])
Diabetes.automf(3, names=['Rendah', 'Sedang', 'Tinggi'])

# Rules
rule1 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Rendah'] & umur['Rendah'], Diabetes['Rendah'])
rule2 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Rendah'] & umur['Rendah'], Diabetes['Sedang'])
rule3 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Rendah'] & umur['Sedang'], Diabetes['Sedang'])
rule4 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'] & umur['Rendah'], Diabetes['Sedang'])
rule5 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'] & umur['Sedang'], Diabetes['Sedang'])
rule6 = ctrl.Rule(kadar_gula_darah['Tinggi'] & BMI['Tinggi'] & umur['Tinggi'], Diabetes['Tinggi'])
rule7 = ctrl.Rule(kadar_gula_darah['Tinggi'] & BMI['Tinggi'], Diabetes['Tinggi'])
rule8 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'], Diabetes['Sedang'])
rule9 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'], Diabetes['Tinggi'])
rule10 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Sedang'], Diabetes['Sedang'])
rule11 = ctrl.Rule(kadar_gula_darah['Sedang'] & umur['Tinggi'], Diabetes['Tinggi'])
rule12 = ctrl.Rule(kadar_gula_darah['Rendah'] & umur['Tinggi'] & BMI['Rendah'], Diabetes['Sedang'])
rule13 = ctrl.Rule(kadar_gula_darah['Tinggi'] & umur['Rendah'], Diabetes['Sedang'])
rule14 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Rendah'] & umur['Rendah'], Diabetes['Rendah'])

# Control System
diabetes_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14])
diabetes_sim = ctrl.ControlSystemSimulation(diabetes_ctrl)

def kategorisasi_risiko(skor):
    """Mengubah skor numerik 0-100 menjadi kategori verbal."""
    if skor <= 33.33:
        return "Rendah"
    elif skor <= 66.66:
        return "Sedang"
    else:
        return "Tinggi"

def prediksi_single(input_bmi, input_umur, input_gula):
    """Fungsi untuk memprediksi satu data individu."""
    try:
        # Clipping data agar aman
        b = np.clip(input_bmi, BMI_min, BMI_max)
        u = np.clip(input_umur, umur_min, umur_max)
        g = np.clip(input_gula, kadar_gula_darah_min, kadar_gula_darah_max)
        
        diabetes_sim.input['BMI'] = b
        diabetes_sim.input['umur'] = u
        diabetes_sim.input['kadar_gula_darah'] = g
        
        diabetes_sim.compute()
        hasil_skor = diabetes_sim.output['pred_diabetes']
        kategori = kategorisasi_risiko(hasil_skor)
        return hasil_skor, kategori
    except Exception as e:
        return 0, "Error/Tidak Ada Rule Cocok"

print("Sedang memproses data dari CSV...")
# Batasi 100 data dulu agar cepat untuk testing
df_run = df.head(100).copy() 

hasil_skor_list = []
hasil_kategori_list = []

for idx, row in df_run.iterrows():
    skor, kat = prediksi_single(row['bmi'], row['age'], row['blood_glucose_level'])
    hasil_skor_list.append(skor)
    hasil_kategori_list.append(kat)

df_run['prediksi_skor'] = hasil_skor_list
df_run['kategori_risiko'] = hasil_kategori_list

print("\n=== HASIL PREDIKSI (10 Data Pertama) ===")
print(df_run[['bmi', 'age', 'blood_glucose_level', 'prediksi_skor', 'kategori_risiko']].head(10))

# Visualisasi
plt.figure(figsize=(8, 5))
counts = df_run['kategori_risiko'].value_counts()
colors = {'Rendah':'green', 'Sedang':'orange', 'Tinggi':'red'}
plt.bar(counts.index, counts.values, color=[colors.get(x, 'blue') for x in counts.index])
plt.title('Distribusi Risiko Diabetes (Dari Data CSV)')
plt.xlabel('Kategori Risiko')
plt.ylabel('Jumlah Orang')
plt.show()