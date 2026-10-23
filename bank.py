import pandas as pd
import numpy as np  # Ditambahkan untuk Data Cleaning
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. DATA & LOADING (Tugas Data Mining)
# =========================================================
# Referensi Metode Loading Data di Pandas (Pandas.pydata.org):
# - pd.read_csv('file.csv')
# - pd.read_excel('file.xlsx')
# - pd.read_json('file.json')
# - pd.read_html('file.html')
# - pd.read_xml('file.xml')
# - pd.read_sql('query', connection)

file_path = "bank-full.csv"
df = pd.read_csv(file_path, sep=";")

print("=" * 60)
print("1. DATA & LOADING (KAMPANYE DEPOSITO BANK)")
print("=" * 60)

print("Jumlah data awal :", len(df))
print("Jumlah kolom     :", len(df.columns))

# Pengecekan Syarat Minimal 1000 Data 
if len(df) >= 1000:
    print(f"[OK] Syarat Dataset: Memenuhi ketentuan (Jumlah data = {len(df)} >= 1000)")
else:
    print(f"[INFO] Syarat Dataset: Kurang dari 1000 data (Jumlah data = {len(df)})")

print("\n5 Data Pertama:")
print(df.head())


# =========================================================
# 2. EKSPLORASI DATA (Catatan Tugas Data Mining)
# =========================================================
print("\n" + "=" * 60)
print("2. EKSPLORASI DATA")
print("=" * 60)

# Poin 1: Pembagian Jenis Data (Kategorikal & Numerikal)
kolom_numerik = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
kolom_kategori = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("\n[Poin 1] Pembagian Jenis Data:")
print(f"• Kolom Numerikal ({len(kolom_numerik)})   :", kolom_numerik)
print(f"• Kolom Kategorikal ({len(kolom_kategori)}) :", kolom_kategori)

# Poin 2: Penjelasan Tipe Data di Python / Pandas
print("\n[Poin 2] Penjelasan Tipe Data di Pandas:")
print("• Data Kategorikal  : Menggunakan tipe 'object' atau 'category' (bukan float/void).")
print("• Data Numerikal    : Menggunakan tipe 'int64' (integer) atau 'float64' (desimal).")

# Poin 3: Memeriksa Jumlah Duplicate & Null pada Dataset
jumlah_duplikat = df.duplicated().sum()
jumlah_null = df.isnull().sum()

print("\n[Poin 3] Cek Duplikat & Data Kosong (Null):")
print(f"• Jumlah Data Duplikat : {jumlah_duplikat} baris")
print("• Jumlah Data Null per Kolom (Sebelum Cleaning):")
print(jumlah_null)


# =========================================================
# 2.5. PROSES DATA CLEANING
# =========================================================
print("\n" + "=" * 60)
print("PROSES DATA CLEANING")
print("=" * 60)

# 1. Menghapus data duplikat jika ada
if jumlah_duplikat > 0:
    print(f"Ditemukan {jumlah_duplikat} baris data duplikat. Menghapus duplikat...")
    df = df.drop_duplicates()
else:
    print("Tidak ditemukan data duplikat.")

# 2. Menangani nilai 'unknown' sebagai missing values (NaN)
print("Mengubah nilai 'unknown' menjadi NaN (Kosong)...")
df.replace("unknown", np.nan, inplace=True)

# Cek kembali jumlah data kosong sesungguhnya setelah "unknown" diubah
print("\nJumlah data kosong (setelah 'unknown' dikonversi):")
print(df.isnull().sum())

# 3. Menghapus baris yang mengandung data kosong
print("\nMenghapus baris yang memiliki data kosong...")
df = df.dropna()

print(f"Jumlah data setelah proses cleaning selesai: {len(df)}")

# Simpan data bersih ke file CSV baru (Opsional)
df.to_csv("bank_cleaned.csv", index=False, sep=";")
print("Data bersih berhasil disimpan ke 'bank_cleaned.csv'")


# =========================================================
# 3. ANALISIS TARGET (Apakah nasabah membuka deposito?)
# =========================================================
print("\n" + "=" * 60)
print("3. ANALISIS TARGET (Buka Deposito / y)")
print("=" * 60)

print(df["y"].value_counts())

print("\nPersentase:")
print(
    df["y"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# =========================================================
# 4. GRAFIK STATUS TARGET
# =========================================================
plt.figure(figsize=(7, 5))

df["y"].value_counts().plot(
    kind="bar",
    color=['skyblue', 'salmon']
)

plt.title("Status Nasabah Membuka Deposito (Target: y)")
plt.xlabel("Buka Deposito")
plt.ylabel("Jumlah Nasabah")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# 5. GRAFIK DISTRIBUSI UMUR NASABAH
# =========================================================
plt.figure(figsize=(8, 5))

plt.hist(
    df["age"],
    bins=15,
    color='lightgreen',
    edgecolor='black'
)

plt.title("Distribusi Umur Nasabah")
plt.xlabel("Umur")
plt.ylabel("Jumlah Nasabah")

plt.tight_layout()
plt.show()


# =========================================================
# 6. ANALISIS BERDASARKAN PEKERJAAN
# =========================================================
pekerjaan = pd.crosstab(
    df["job"],
    df["y"]
)

print("\n" + "=" * 60)
print("STATUS DEPOSITO BERDASARKAN PEKERJAAN")
print("=" * 60)

print(pekerjaan)

pekerjaan.plot(
    kind="bar",
    figsize=(10, 5),
    stacked=False
)

plt.title("Status Buka Deposito Berdasarkan Pekerjaan")
plt.xlabel("Pekerjaan (Job)")
plt.ylabel("Jumlah Nasabah")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# =========================================================
# 7. PREPROCESSING & ENCODING
# =========================================================
data = df.copy()

# Mengambil semua kolom yang bertipe teks/kategori (object) selain target 'y'
kolom_kategori_proses = data.select_dtypes(include=['object', 'category']).columns.tolist()
if 'y' in kolom_kategori_proses:
    kolom_kategori_proses.remove('y')

encoder = LabelEncoder()

for kolom in kolom_kategori_proses:
    data[kolom] = encoder.fit_transform(data[kolom].astype(str))

# Encoding target 'y'
target_encoder = LabelEncoder()
data["y"] = target_encoder.fit_transform(data["y"].astype(str))


# =========================================================
# 8. MEMISAHKAN FITUR DAN TARGET
# =========================================================
X = data.drop(columns=["y"])
y = data["y"]

print("\n" + "=" * 60)
print("FITUR YANG DIGUNAKAN")
print("=" * 60)

print(X.columns.tolist())


# =========================================================
# 9. TRAINING DAN TESTING
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nJumlah data training :", len(X_train))
print("Jumlah data testing  :", len(X_test))


# =========================================================
# 10. DECISION TREE
# =========================================================
decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree.fit(X_train, y_train)
pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, pred_tree)

print("\n" + "=" * 60)
print("HASIL DECISION TREE")
print("=" * 60)

print("Akurasi:", round(accuracy_tree * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, pred_tree))


# =========================================================
# 11. CONFUSION MATRIX DECISION TREE
# =========================================================
cm_tree = confusion_matrix(y_test, pred_tree)

plt.figure(figsize=(6, 5))
plt.imshow(cm_tree, cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.colorbar()

for i in range(len(cm_tree)):
    for j in range(len(cm_tree[i])):
        plt.text(
            j, i, cm_tree[i, j],
            ha="center", va="center",
            color="white" if cm_tree[i, j] > (cm_tree.max() / 2) else "black"
        )

plt.tight_layout()
plt.show()


# =========================================================
# 12. RANDOM FOREST
# =========================================================
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)
pred_forest = random_forest.predict(X_test)
accuracy_forest = accuracy_score(y_test, pred_forest)

print("\n" + "=" * 60)
print("HASIL RANDOM FOREST")
print("=" * 60)

print("Akurasi:", round(accuracy_forest * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, pred_forest))


# =========================================================
# 13. CONFUSION MATRIX RANDOM FOREST
# =========================================================
cm_forest = confusion_matrix(y_test, pred_forest)

plt.figure(figsize=(6, 5))
plt.imshow(cm_forest, cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.colorbar()

for i in range(len(cm_forest)):
    for j in range(len(cm_forest[i])):
        plt.text(
            j, i, cm_forest[i, j],
            ha="center", va="center",
            color="white" if cm_forest[i, j] > (cm_forest.max() / 2) else "black"
        )

plt.tight_layout()
plt.show()


# =========================================================
# 14. PERBANDINGAN AKURASI
# =========================================================
model_names = ["Decision Tree", "Random Forest"]
accuracies = [accuracy_tree, accuracy_forest]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e'])
plt.title("Perbandingan Akurasi Model Klasifikasi")
plt.xlabel("Algoritma")
plt.ylabel("Akurasi")
plt.ylim(0, 1)

for i, value in enumerate(accuracies):
    plt.text(
        i, value + 0.02,
        f"{value * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# =========================================================
# 15. FEATURE IMPORTANCE
# =========================================================
importance = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": random_forest.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=True)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print(importance.sort_values(by="Importance", ascending=False))

plt.figure(figsize=(10, 8))
plt.barh(importance["Fitur"], importance["Importance"], color='teal')
plt.title("Feature Importance - Random Forest (Bank Dataset)")
plt.xlabel("Tingkat Kepentingan (Importance)")
plt.ylabel("Fitur")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("PROGRAM SELESAI")
print("=" * 60)