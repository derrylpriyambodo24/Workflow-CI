import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# 1. Load Dataset
# Pastikan nama file sesuai
df = pd.read_csv('water_potability_preprocessing.csv')

# 2. Pisahkan Fitur dan Target
# PERBAIKAN DI SINI: Gunakan 'Potability' sebagai nama kolom yang didrop dan diambil
X = df.drop(columns=['Potability'])
y = df['Potability']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Set Eksperimen & Autolog
mlflow.set_experiment("Eksperimen_Basic_Derryl")

with mlflow.start_run():
    # Aktifkan log_models=True
    mlflow.sklearn.autolog(log_models=True)

    # 4. Training Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 5. Evaluasi
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
