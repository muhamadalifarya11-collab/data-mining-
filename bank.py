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
# 1. MEMBACA DATASET
# =========================================================
# Pastikan pemisah kolom sesuai dengan file Anda (biasanya koma ',' atau titik koma ';')
df = pd.read_csv("bank-full.csv", sep=";")

print("=" * 60)
print("ANALISIS DATA KAMPANYE DEPOSITO BANK")
print("=" * 60)

print("Jumlah data awal :", len(df))
print("Jumlah kolom     :", len(df.columns))

print("\n5 Data Pertama:")
print(df.head())


# =========================================================
# 2. CEK DATA AWAL
# =========================================================untuk cleaning

print("\n" + "=" * 60)
print("CEK DATA KOSONG (SEBELUM CLEANING)")
print("=" * 60)

print(df.isnull().sum())


# =========================================================
# 2.5. PROSES DATA CLEANING (TAMBAHAN)
# =========================================================

print("\n" + "=" * 60)
print("PROSES DATA CLEANING")
print("=" * 60)

# 1. Menghapus data duplikat (jika ada)
jumlah_duplikat = df.duplicated().sum()
print(f"Ditemukan {jumlah_duplikat} baris data duplikat. Menghapus duplikat...")
df = df.drop_duplicates()

# 2. Menangani nilai "unknown" yang sebenarnya adalah missing values
print("Mengubah nilai 'unknown' menjadi NaN (Kosong)...")
df.replace("unknown", np.nan, inplace=True)

# Cek kembali jumlah data kosong sesungguhnya setelah "unknown" diubah
print("\nJumlah data kosong (setelah 'unknown' dikonversi):")
print(df.isnull().sum())

# 3. Menghapus baris yang mengandung data kosong
print("\nMenghapus baris yang memiliki data kosong...")
df = df.dropna()

print(f"Jumlah data setelah proses cleaning selesai: {len(df)}")


# =========================================================
# 3. ANALISIS TARGET (Apakah nasabah membuka deposito?)
# =========================================================

print("\n" + "=" * 60)
print("DISTRIBUSI TARGET (Buka Deposito / y)")
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
# 7. PREPROCESSING
# =========================================================

# Kita copy data yang SUDAH BERSIH
data = df.copy()


# =========================================================
# 8. ENCODING DATA KATEGORI
# =========================================================

# Mengambil semua kolom yang bertipe teks/kategori (object) selain target 'y'
kolom_kategori = data.select_dtypes(include=['object']).columns.tolist()
if 'y' in kolom_kategori:
    kolom_kategori.remove('y')

encoder = LabelEncoder()

for kolom in kolom_kategori:
    data[kolom] = encoder.fit_transform(data[kolom])

# Encoding target 'y'
target_encoder = LabelEncoder()
data["y"] = target_encoder.fit_transform(data["y"])


# =========================================================
# 9. MEMISAHKAN FITUR DAN TARGET
# =========================================================

X = data.drop(columns=["y"])
y = data["y"]

print("\n" + "=" * 60)
print("FITUR YANG DIGUNAKAN")
print("=" * 60)

print(X.columns.tolist())


# =========================================================
# 10. TRAINING DAN TESTING
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
# 11. DECISION TREE
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
# 12. CONFUSION MATRIX DECISION TREE
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
# 13. RANDOM FOREST
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
# 14. CONFUSION MATRIX RANDOM FOREST
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
# 15. PERBANDINGAN AKURASI
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
# 16. FEATURE IMPORTANCE
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