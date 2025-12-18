import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


file_path = "diabetes_cleaned_with_label.csv"   
df = pd.read_csv(file_path)
print(df.head())
print(df)


BMI_min = int(df['bmi'].min()-5)
BMI_max = int(df['bmi'].max()+5)
umur_min = int(df['age'].min()-5)
umur_max = int(df['age'].max()+5)
kadar_gula_darah_min = int(df['blood_glucose_level'].min()-5)
kadar_gula_darah_max = int(df['blood_glucose_level'].max()+5)

BMI = ctrl.Antecedent(np.arange(BMI_min,BMI_max + 1,1), 'BMI')
umur = ctrl.Antecedent(np.arange(umur_min,umur_max + 1,1), 'umur')
kadar_gula_darah = ctrl.Antecedent(np.arange(kadar_gula_darah_min,kadar_gula_darah_max + 1,1), 'kadar_gula_darah')
Diabetes = ctrl.Consequent(np.arange(0,100,1), 'pred_diabetes')

BMI.automf(3, names=['Rendah','Sedang','Tinggi'])
umur.automf(3,names=['Rendah','Sedang','Tinggi'])
kadar_gula_darah.automf(3,names=['Rendah','Sedang','Tinggi'])
Diabetes.automf(3,names=['Rendah','Sedang','Tinggi'])

rule1 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Rendah'] & umur['Rendah'], Diabetes['Rendah'])
rule2 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Rendah'] & umur['Rendah'], Diabetes['Sedang'])
rule3 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Rendah'] & umur['Sedang'],Diabetes['Sedang'])
rule4 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'] & umur['Rendah'],Diabetes['Sedang'])
rule5 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'] & umur['Sedang'],Diabetes['Sedang'])
rule6 = ctrl.Rule(kadar_gula_darah['Tinggi'] & BMI ['Tinggi'] & umur['Tinggi'], Diabetes['Tinggi'])
rule7 = ctrl.Rule(kadar_gula_darah['Tinggi'] & BMI['Tinggi'], Diabetes['Tinggi'])
rule8 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'],Diabetes['Sedang'])
rule9 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Tinggi'],Diabetes['Tinggi'])
rule10 = ctrl.Rule(kadar_gula_darah['Rendah'] & BMI['Sedang'], Diabetes ['Sedang'])
rule11 = ctrl.Rule(kadar_gula_darah['Sedang'] & umur['Tinggi'], Diabetes['Tinggi'])
rule12 = ctrl.Rule(kadar_gula_darah['Rendah'] & umur['Tinggi'] & BMI['Rendah'], Diabetes['Sedang'])
rule13 = ctrl.Rule(kadar_gula_darah['Tinggi'] & umur['Rendah'],Diabetes['Sedang'])
rule14 = ctrl.Rule(kadar_gula_darah['Sedang'] & BMI['Rendah'] & umur['Rendah'], Diabetes['Rendah'])

rules = [rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14]

diabetes_ctrl = ctrl.ControlSystem(rules)

diabetes_sim = ctrl.ControlSystemSimulation(diabetes_ctrl)

df['prediksi_diabetes'] = 0.0

for idx, row in df.iterrows():
    diabetes_sim = ctrl.ControlSystemSimulation(diabetes_ctrl)

    diabetes_sim.input['BMI'] = row ['bmi']
    diabetes_sim.input['umur'] = row ['age']
    diabetes_sim.input['kadar_gula_darah'] = row ['blood_glucose_level']
   
    
    diabetes_sim.compute()
    
    df.at[idx, 'prediksi_diabetes'] = diabetes_sim.output['pred_diabetes']


df['error'] = df['pred_diabetes'] - df['prediksi_diabetes']
df['absolute_error'] = np.abs(df['error'])
df['squared_error'] = df['error']**2

mae = df['absolute_error'].mean()  
mse = df['squared_error'].mean()   
rmse = np.sqrt(mse) 

print(f"\nMetrik Evaluasi:")
print(f"  - MAE (Mean Absolute Error)  : {mae:.2f}")
print(f"  - MSE (Mean Squared Error)   : {mse:.2f}")
print(f"  - RMSE (Root Mean Squared Error): {rmse:.2f}")

print(f"\nHasil Prediksi (10 data pertama):")
print(df[['pred_diabetes']].head(10).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(df['pred_diabetes'], df['prediksi_diabetes'], alpha=0.6, s=100)
axes[0, 0].plot([0, 100], [0, 100], 'r--', label='Prediksi Perfect')
axes[0, 0].set_xlabel('Kualitas Data Aktual', fontsize=11)
axes[0, 0].set_ylabel('Kualitas Prediksi', fontsize=11)
axes[0, 0].set_title('Perbandingan: data aktual vs Prediksi', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(df['error'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0, 1].set_xlabel('Error (Aktual - Prediksi)', fontsize=11)
axes[0, 1].set_ylabel('Frekuensi', fontsize=11)
axes[0, 1].set_title('Distribusi Error', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df.index, df['kualitas'], marker='o', label='Aktual', linewidth=2)
axes[1, 0].plot(df.index, df['prediksi_kualitas'], marker='s', label='Prediksi', linewidth=2)
axes[1, 0].set_xlabel('Index Data', fontsize=11)
axes[1, 0].set_ylabel('Kualitas', fontsize=11)
axes[1, 0].set_title('Trend: Aktual vs Prediksi', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

print(f"  - Prediksi gejala diabetes: {diabetes_sim.output['pred_diabetes']:.2f}")


plt.show()
