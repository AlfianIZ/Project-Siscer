import pandas as pd
from sklearn.model_selection import train_test_split

# 1. LOAD DATASET
file_path = "diabetes_prediction_dataset.csv"   
df = pd.read_csv(file_path)

# 2. CLEANING DATA
# Hapus missing value
df_clean = df.dropna()

# Lowercase kolom kategorikal
df_clean["gender"] = df_clean["gender"].str.lower()
df_clean["smoking_history"] = df_clean["smoking_history"].str.lower()

# Ubah gender menjadi angka
df_clean["gender"] = df_clean["gender"].map({
    "male": 0,
    "female": 1,
    "other": 2
})

# Label encoding smoking_history
df_clean["smoking_history"] = df_clean["smoking_history"].map({
    "never": 0,
    "former": 1,
    "current": 2,
    "ever": 3,
    "not current": 4,
    "no info": 5
})

# 3. FUZZY LABELING (Risk Low, Medium, High)
def fuzzy_risk_label(row):
    bmi = row["bmi"]
    age = row["age"]
    gula = row["blood_glucose_level"]

    if gula >= 126 or bmi > 30 or age > 60:
        return "High"

    if 100 < gula <= 125 or 25 < bmi <= 30 or 45 < age <= 60:
        return "Medium"

    return "Low"

df_clean["risk_label"] = df_clean.apply(fuzzy_risk_label, axis=1)

# 4. SPLITTING DATA (Train 80%, Test 20%)
X = df_clean[["bmi", "age", "blood_glucose_level"]]
y = df_clean["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== SHAPE DATA ===")
print("Train:", X_train.shape, " Test:", X_test.shape)

print("\n=== DISTRIBUSI LABEL ===")
print(y_train.value_counts())
print(y_test.value_counts())

print("\n=== 5 DATA AWAL DENGAN LABEL ===")
print(df_clean[["bmi", "age", "blood_glucose_level", "risk_label"]].head())

# 5. SIMPAN DATASET HASIL CLEANING + LABELING KE CSV
df_clean.to_csv("diabetes_cleaned_with_label.csv", index=False)
print("\nFile 'diabetes_cleaned_with_label.csv' berhasil dibuat!")

# 6. SIMPAN DATA HASIL SPLITTING KE CSV
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("File splitting (X_train, X_test, y_train, y_test) berhasil dibuat!")
