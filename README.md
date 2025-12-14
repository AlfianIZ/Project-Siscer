# Deskripsi
Disini ada dua source code, pada ver_2.py Jika ada kategori yang tidak terpetakan program akan menampilkan warning, pada cleaning.py tidak ada penanganan untuk hal tersebut. Main program kita menggunakan diabetes_prdiction_fis.py yang berada pada folder kodePerbaikan, pada folder tersebut berisi dataset yang sudah di cleaning, kode cleaning yang diperbaiki dan main program.
## Deskripsi dataset
Dataset dapat diakses di : [kaggle | Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data). 

Gula : Gula darah
BMI : Indeks Massa Tubuh seseorang.

age : umur seseorang.

## NOTE
Jika mau menjalankan program-program tersebut, pastikan sudah menginstal Pandas, Scikit-learn, scikit-fuzzy, . Jika belum maka harus menginstalnya terlebih dahulu. dengan menjalankan perintah berikut pada terminal atau cmd :

```bash
pip install pandas
```
```bash
pip install scikit-learn
```
```bash
pip install scikit-fuzzy
```

# #Sebelum Diperbaiki

## Output 
Output dari cleaning.py :

```bash
=== SHAPE DATA ===
Train: (80000, 3)  Test: (20000, 3)

=== DISTRIBUSI LABEL ===
risk_label
High      65855
Medium     8715
Low        5430
Name: count, dtype: int64
risk_label
High      16463
Medium     2179
Low        1358
Name: count, dtype: int64

=== 5 DATA AWAL DENGAN LABEL ===
     bmi   age  blood_glucose_level risk_label
0  25.19  80.0                  140       High
1  27.32  54.0                   80     Medium
2  27.32  28.0                  158       High
3  23.45  36.0                  155       High
4  20.14  76.0                  155       High
```
Dari output tersebut pada bagian SHAPE DATA, data yang akan ditrain berjumlah 80.000 dan data test sebanyak 20.000. Pada bagian DISTRIBUSI LABEL, risk_label yang atas adalah data train dan yang dibawah data test. Pada bagian 5 DATA AWAL DENGAN LABEL adalah contoh 5 data awal yang ada pada dataset tersebut.

Output dari ver_2.py :

```bash
=== Unique gender values: ['female' 'male' 'other']
=== Unique smoking values: ['never' 'no info' 'current' 'former' 'ever' 'not current']

=== Distribusi label sebelum splitting ===
risk_label
High      79819
Medium    13393
Low        6788
Name: count, dtype: int64

=== SHAPE DATA ===
Train: (80000, 3)  Test: (20000, 3)

=== DISTRIBUSI LABEL TRAIN ===
risk_label
High      63855
Medium    10715
Low        5430
Name: count, dtype: int64

=== DISTRIBUSI LABEL TEST ===
risk_label
High      15964
Medium     2678
Low        1358
Name: count, dtype: int64

=== 5 DATA AWAL ===
     bmi   age  blood_glucose_level risk_label
0  25.19  80.0                  140       High
1  27.32  54.0                   80     Medium
2  27.32  28.0                  158       High
3  23.45  36.0                  155       High
4  20.14  76.0                  155       High
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# #Setelah Diperbaiki

## Output 
Output dari diabetes_prdiction_fis.py:

```
Mencoba memuat data bersih...
Berhasil! Memproses SELURUH DATA (96128 baris) dengan Fuzzy Logic...
Mohon tunggu, proses ini mungkin memakan waktu agak lama...
Sedang memproses baris ke-5000...
Sedang memproses baris ke-10000...
Sedang memproses baris ke-15000...
Sedang memproses baris ke-20000...
Sedang memproses baris ke-25000...
Sedang memproses baris ke-30000...
Sedang memproses baris ke-35000...
Sedang memproses baris ke-40000...
Sedang memproses baris ke-45000...
Sedang memproses baris ke-50000...
Sedang memproses baris ke-55000...
Sedang memproses baris ke-60000...
Sedang memproses baris ke-65000...
Sedang memproses baris ke-70000...
Sedang memproses baris ke-75000...
Sedang memproses baris ke-80000...
Sedang memproses baris ke-85000...
Sedang memproses baris ke-90000...
Sedang memproses baris ke-95000...
Proses Fuzzy selesai!

========================================
 AKURASI SISTEM (TOTAL): 78.65%
========================================
Menampilkan grafik distribusi...
```
<img width="1001" height="675" alt="image" src="https://github.com/user-attachments/assets/85407286-de96-4aca-b721-6182044ab7fa" />
<img width="936" height="715" alt="keanggotaan FIS" src="https://github.com/user-attachments/assets/2e66a687-1b05-435d-b549-84a70971a3ea" />


```
--- MODE INPUT MANUAL ---

Tes diagnosa pasien? (y/n): y
BMI : 24
Umur: 20
Gula: 135

>>> HASIL: Sedang (Skor: 50.00)
Menampilkan grafik keanggotaan...
```

<img width="1002" height="873" alt="image" src="https://github.com/user-attachments/assets/a627a48d-4c6a-4a17-9928-b3367f93ac0e" />

```

Tes diagnosa pasien? (y/n): n
```

------

Output dari cleaning.py yang diperbaiki :
```
=== MULAI PROSES CLEANING & LABELING ===
Data awal dimuat: 100000 baris
Sedang membuat label baru...

========================================
 SUKSES! 
========================================
File bersih tersimpan sebagai: 'diabetes_cleaned.csv'
Total Data Bersih: 96128
Distribusi Label:
Label_Baru
Rendah    39013
Sedang    32008
Tinggi    25107
Name: count, dtype: int64
```
cleaning.py juga menghasilkan output dataset yang sudah dicleaning yaitu diabetes_cleaned.csv, file tersebut ada pada folder kodePerbaikan.
