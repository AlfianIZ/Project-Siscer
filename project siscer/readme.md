# Deskripsi
Disini ada dua source code, pada ver_2.py Jika ada kategori yang tidak terpetakan program akan menampilkan warning, pada cleaning.py tidak ada penanganan untuk hal tersebut.
## Deskripsi dataset
Dataset dapat diakses di : [kaggle | Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data). 

Gula : Gula disini yang dibuat untuk aturan adalah gula darah puasa (diambil setelah tidak makan  semalaman) atau setelah bangun tidur pada pagi hari.

BMI : Indeks Massa Tubuh seseorang.

age : umur seseorang.

# NOTE
Jika mau menjalankan kedua program tersebut, pastikan sudah menginstal Pandas dan Scikit-learn. Jika belum maka harus menginstalnya terlebih dahulu. dengan menjalankan perintah berikut pada terminal atau cmd :

```bash
pip install pandas
```
```bash
pip install scikit-learn
```

# Output 
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