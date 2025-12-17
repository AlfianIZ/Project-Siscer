import pandas as pd
import numpy as np

def proses_cleaning_labeling():
    print("=== MULAI PROSES CLEANING & LABELING ===")
    
    nama_file_input = "diabetes_prediction_dataset.csv"
    nama_file_output = "diabetes_cleaned.csv"

    try:
        df = pd.read_csv(nama_file_input)
        print(f"Data awal dimuat: {len(df)} baris")

        df = df.drop_duplicates()
        df = df[df['gender'] != 'Other']
        
        def label_logika_baru(row):
            gula = row['blood_glucose_level']
            bmi = row['bmi']
            age = row['age']

            if gula >= 200:
                return 'Tinggi'
            if gula >= 155 and (bmi > 30 or age > 50):
                return 'Tinggi'
            if 140 <= gula < 200:
                return 'Sedang'
            if 100 <= gula < 140 and (bmi > 30 and age > 45):
                return 'Sedang'
 
            return 'Rendah'

        print("Sedang membuat label baru...")
        df['Label_Baru'] = df.apply(label_logika_baru, axis=1)

        df_final = df[['gender', 'age', 'bmi', 'blood_glucose_level', 'Label_Baru']]
        df_final.to_csv(nama_file_output, index=False)
        
        print("\n" + "="*40)
        print(" SUKSES! ")
        print("="*40)
        print(f"File bersih tersimpan sebagai: '{nama_file_output}'")
        print(f"Total Data Bersih: {len(df_final)}")
        print(f"Distribusi Label:\n{df_final['Label_Baru'].value_counts()}")
        
    except FileNotFoundError:
        print(f"ERROR: File '{nama_file_input}' tidak ditemukan.")

if __name__ == "__main__":

    proses_cleaning_labeling()
