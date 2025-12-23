import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Cek apakah file dataset ada
if not os.path.exists("water_potability_preprocessing.csv"):
    # Fallback jika file ada di folder root tapi script dijalankan dari folder lain, atau sebaliknya
    # Namun untuk CI sederhana, biasanya file ada di root.
    print("Warning: File water_potability_preprocessing.csv tidak ditemukan di path saat ini.")

try:
    df = pd.read_csv("water_potability_preprocessing.csv")
    
    # Preprocessing sederhana
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Setup MLflow
    mlflow.set_experiment("CI_Eksperimen_Air")
    mlflow.sklearn.autolog()

    # Training
    print("Mulai Training...")
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model trained with Accuracy: {acc}")

except Exception as e:
    print(f"Terjadi Error: {e}")
    # Raise error agar CI statusnya menjadi Failure jika ada masalah
    raise e
