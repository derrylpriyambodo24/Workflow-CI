import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# 1. Load Dataset
# Pastikan file ada di satu folder dengan script ini
df = pd.read_csv('water_potability_preprocessing.csv')

# 2. Pisahkan Fitur dan Target
# Menggunakan kolom 'Potability' sesuai dataset kamu
X = df.drop(columns=['Potability'])
y = df['Potability']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PERBAIKAN UTAMA ADA DI BAWAH INI ---

# JANGAN gunakan set_experiment saat dijalankan via 'mlflow run' di CI/CD
# karena akan bentrok dengan Run ID yang dibuat otomatis.
# mlflow.set_experiment("Eksperimen_Basic_Derryl")  <-- INI DIHAPUS/KOMENTAR

# Aktifkan autolog
mlflow.sklearn.autolog(log_models=True)

# Mulai Run (akan otomatis menggunakan Run ID dari GitHub Actions)
with mlflow.start_run():
    # 3. Training Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. Evaluasi
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
