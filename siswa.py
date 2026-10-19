import pandas as pd
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

df = pd.read_csv("dataset_siswa_greget_1200.csv")

print("=" * 60)
print("ANALISIS DATA SISWA SMK")
print("=" * 60)

print("Jumlah data   :", len(df))
print("Jumlah kolom  :", len(df.columns))

print("\n5 Data Pertama:")
print(df.head())


# =========================================================
# 2. CEK DATA
# =========================================================

print("\n" + "=" * 60)
print("CEK DATA KOSONG")
print("=" * 60)

print(df.isnull().sum())


# =========================================================
# 3. ANALISIS TARGET
# =========================================================

print("\n" + "=" * 60)
print("DISTRIBUSI TARGET")
print("=" * 60)

print(df["Lanjut_Kuliah"].value_counts())

print("\nPersentase:")
print(
    df["Lanjut_Kuliah"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# =========================================================
# 4. GRAFIK STATUS LANJUT KULIAH
# =========================================================

plt.figure(figsize=(7, 5))

df["Lanjut_Kuliah"].value_counts().plot(
    kind="bar"
)

plt.title("Status Siswa Melanjutkan Kuliah")
plt.xlabel("Lanjut Kuliah")
plt.ylabel("Jumlah Siswa")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# 5. GRAFIK NILAI RATA-RATA
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Nilai_Rata_Rata"],
    bins=10
)

plt.title("Distribusi Nilai Rata-Rata Siswa")
plt.xlabel("Nilai Rata-Rata")
plt.ylabel("Jumlah Siswa")

plt.tight_layout()
plt.show()


# =========================================================
# 6. ANALISIS BERDASARKAN JURUSAN
# =========================================================

jurusan = pd.crosstab(
    df["Jurusan"],
    df["Lanjut_Kuliah"]
)

print("\n" + "=" * 60)
print("STATUS KULIAH BERDASARKAN JURUSAN")
print("=" * 60)

print(jurusan)

jurusan.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Status Lanjut Kuliah Berdasarkan Jurusan")
plt.xlabel("Jurusan")
plt.ylabel("Jumlah Siswa")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# 7. PREPROCESSING
# =========================================================

data = df.copy()

# ID tidak digunakan dalam klasifikasi
data = data.drop(
    columns=["ID_Siswa"]
)


# =========================================================
# 8. ENCODING DATA KATEGORI
# =========================================================

kolom_kategori = [
    "Jenis_Kelamin",
    "Jurusan",
    "Pendapatan_Orang_Tua",
    "Dukungan_Orang_Tua",
    "Minat_Belajar",
    "Rencana_Karier",
    "Jarak_Kampus"
]

encoder = LabelEncoder()

for kolom in kolom_kategori:

    data[kolom] = encoder.fit_transform(
        data[kolom]
    )


# Encoding target
target_encoder = LabelEncoder()

data["Lanjut_Kuliah"] = target_encoder.fit_transform(
    data["Lanjut_Kuliah"]
)


# =========================================================
# 9. MEMISAHKAN FITUR DAN TARGET
# =========================================================

X = data.drop(
    columns=["Lanjut_Kuliah"]
)

y = data["Lanjut_Kuliah"]


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

decision_tree.fit(
    X_train,
    y_train
)

pred_tree = decision_tree.predict(
    X_test
)

accuracy_tree = accuracy_score(
    y_test,
    pred_tree
)


print("\n" + "=" * 60)
print("HASIL DECISION TREE")
print("=" * 60)

print(
    "Akurasi:",
    round(accuracy_tree * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred_tree
    )
)


# =========================================================
# 12. CONFUSION MATRIX DECISION TREE
# =========================================================

cm_tree = confusion_matrix(
    y_test,
    pred_tree
)

plt.figure(figsize=(6, 5))

plt.imshow(cm_tree)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")

plt.colorbar()

for i in range(len(cm_tree)):
    for j in range(len(cm_tree[i])):

        plt.text(
            j,
            i,
            cm_tree[i, j],
            ha="center",
            va="center"
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

random_forest.fit(
    X_train,
    y_train
)

pred_forest = random_forest.predict(
    X_test
)

accuracy_forest = accuracy_score(
    y_test,
    pred_forest
)


print("\n" + "=" * 60)
print("HASIL RANDOM FOREST")
print("=" * 60)

print(
    "Akurasi:",
    round(accuracy_forest * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred_forest
    )
)


# =========================================================
# 14. CONFUSION MATRIX RANDOM FOREST
# =========================================================

cm_forest = confusion_matrix(
    y_test,
    pred_forest
)

plt.figure(figsize=(6, 5))

plt.imshow(cm_forest)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")

plt.colorbar()

for i in range(len(cm_forest)):
    for j in range(len(cm_forest[i])):

        plt.text(
            j,
            i,
            cm_forest[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# =========================================================
# 15. PERBANDINGAN AKURASI
# =========================================================

model_names = [
    "Decision Tree",
    "Random Forest"
]

accuracies = [
    accuracy_tree,
    accuracy_forest
]

plt.figure(figsize=(8, 5))

plt.bar(
    model_names,
    accuracies
)

plt.title("Perbandingan Akurasi Model")
plt.xlabel("Algoritma")
plt.ylabel("Akurasi")

plt.ylim(0, 1)

for i, value in enumerate(accuracies):

    plt.text(
        i,
        value + 0.02,
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

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance)


plt.figure(figsize=(10, 6))

plt.barh(
    importance["Fitur"],
    importance["Importance"]
)

plt.title(
    "Feature Importance - Random Forest"
)

plt.xlabel("Importance")
plt.ylabel("Fitur")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


print("\n" + "=" * 60)
print("PROGRAM SELESAI")
print("=" * 60)