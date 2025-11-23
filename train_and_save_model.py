import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Ruta al CSV 
DATA_PATH = "flight_delays_hackathon.csv"

df = pd.read_csv(DATA_PATH)

# Ajusta nombre de objetivo si difiere
TARGET = "departure_delay"  

# Limpieza simple
df = df.drop_duplicates().dropna()

# Features y target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identifica columnas 
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

# ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Pipeline completo
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar
pipeline.fit(X_train, y_train)

# Evaluacion sencilla
r2 = pipeline.score(X_test, y_test)
print(f"R2 en test: {r2:.4f}")

# Guardar pipeline
OUT_DIR = "model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
joblib.dump(pipeline, os.path.join(OUT_DIR, "rf_pipeline.joblib"))
print("Pipeline guardado en:", os.path.join(OUT_DIR, "rf_pipeline.joblib"))