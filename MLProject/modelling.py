import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# 1. Load Dataset
# Pastikan nama file sesuai dengan yang ada di folder MLProject
df = pd.read_csv('water_potability_preprocessing.csv') 

# Pisahkan fitur dan target (Sesuaikan 'Target' dengan nama kolom label kamu)
X = df.drop(columns=['Target']) 
y = df['Target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Set Eksperimen & Autolog
mlflow.set_experiment("Eksperimen_Basic_Derryl")

with mlflow.start_run():
    # PENTING: Aktifkan log_models=True sesuai request reviewer
    mlflow.sklearn.autolog(log_models=True)

    # 3. Training Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. Evaluasi (Metrics akan otomatis ter-log oleh autolog, tapi print juga oke)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
