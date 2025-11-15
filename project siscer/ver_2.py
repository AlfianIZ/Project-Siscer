import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# 1. LOAD DATASET SECARA AMAN
# ============================================================

file_path = "diabetes_prediction_dataset.csv"   # pastikan file ada di folder yang sama
df = pd.read_csv(file_path)

# Buat salinan agar tidak memicu SettingWithCopyWarning
df_clean = df.copy()

# ============================================================
# 2. CLEANING DATA
# ============================================================

# Hapus baris dengan missing values
df_clean = df_clean.dropna().copy()

# Pastikan kolom yang akan di-lowercase bertipe string
df_clean["gender"] = df_clean["gender"].astype(str).str.lower()
df_clean["smoking_history"] = df_clean["smoking_history"].astype(str).str.lower()

# ============================================================
# 3. VALIDASI NILAI UNIK (MENINGKATKAN KEAMANAN)
# ============================================================

print("=== Unique gender values:", df_clean["gender"].unique())
print("=== Unique smoking values:", df_clean["smoking_history"].unique())

# ============================================================
# 4. ENCODING KATEGORIKAL
# ============================================================

gender_map = {"male": 0, "female": 1, "other": 2}
smoke_map = {
    "never": 0, "former": 1, "current": 2,
    "ever": 3, "not current": 4, "no info": 5
}

df_clean["gender"] = df_clean["gender"].map(gender_map)
df_clean["smoking_history"] = df_clean["smoking_history"].map(smoke_map)

# Jika ada kategori yang tidak terpetakan → tampilkan
if df_clean["gender"].isna().any():
    print("\n[WARNING] Ada nilai gender yang tidak dikenali!")

if df_clean["smoking_history"].isna().any():
    print("\n[WARNING] Ada nilai smoking_history yang tidak dikenali!")

# ============================================================
# 5. FUZZY LABELING (ATURAN YANG KITA SETTING)
# ============================================================

def fuzzy_risk_label(row):
    bmi = row["bmi"]
    age = row["age"]
    gula = row["blood_glucose_level"]

    # Risiko Tinggi
    if gula >= 126 or bmi > 35 or age > 60:
        return "High"

    # Risiko Sedang
    if (100 < gula <= 125) or (25 < bmi <= 35) or (45 < age <= 60):
        return "Medium"

    # Risiko Rendah
    return "Low"

df_clean["risk_label"] = df_clean.apply(fuzzy_risk_label, axis=1)

# ============================================================
# 6. SPLITTING DATASET
# ============================================================

X = df_clean[["bmi", "age", "blood_glucose_level"]]
y = df_clean["risk_label"]

# Stratify aman hanya jika setiap label punya minimal 2 data
print("\n=== Distribusi label sebelum splitting ===")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 7. OUTPUT RINGKAS
# ============================================================

print("\n=== SHAPE DATA ===")
print("Train:", X_train.shape, " Test:", X_test.shape)

print("\n=== DISTRIBUSI LABEL TRAIN ===")
print(y_train.value_counts())

print("\n=== DISTRIBUSI LABEL TEST ===")
print(y_test.value_counts())

print("\n=== 5 DATA AWAL ===")
print(df_clean[["bmi", "age", "blood_glucose_level", "risk_label"]].head())
