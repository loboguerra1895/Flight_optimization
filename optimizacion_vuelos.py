# ==========================================
# 1. IMPORTACIONES
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. CARGAR DATASET
# ==========================================
df = pd.read_csv("flight_delays_hackathon.csv")

print("Primeras filas del dataset:")
print(df.head())
print("\nResumen del dataset:")
print(df.info())

# ==========================================
# 3. LIMPIEZA BÁSICA
# ==========================================

# 3.1. Eliminar duplicados
df = df.drop_duplicates()

# 3.2. Eliminar filas con valores nulos
df = df.dropna()

# --------------------
# IMPORTANTE:
# Cambiar esta columna por el nombre real del delay en tu dataset
# --------------------
target_column = "departure_delay"  

if target_column not in df.columns:
    raise ValueError(f"La columna objetivo '{target_column}' no existe. Verifica el nombre real en df.columns")


# ==========================================
# 4. PREPARAR VARIABLES X (features) Y y (target)
# ==========================================
X = df.drop(columns=[target_column])
y = df[target_column]

# Convertir columnas categóricas a numéricas
X = pd.get_dummies(X, drop_first=True)

# ==========================================
# 5. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Estandarización SOLO para Regresión Lineal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. MODELOS
# ==========================================

# --------- 6.1. REGRESIÓN LINEAL ----------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# --------- 6.2. ÁRBOL DE DECISIÓN ----------
tree_model = DecisionTreeRegressor(max_depth=8, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# --------- 6.3. RANDOM FOREST ----------
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ==========================================
# 7. EVALUACIÓN
# ==========================================

def evaluate_model(name, true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    print(f"\n=== {name} ===")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

print("RESULTADOS DE LOS MODELOS:")
evaluate_model("Regresión Lineal", y_test, lr_pred)
evaluate_model("Árbol de Decisión", y_test, tree_pred)
evaluate_model("Random Forest", y_test, rf_pred)

numeric_df = df.select_dtypes(include=["int64", "float64"])

numeric_df.hist(figsize=(12, 10))
plt.suptitle("Histogramas de Variables Numéricas", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(
    x=numeric_df.corr()[target_column].index,
    y=numeric_df.corr()[target_column].values
)
plt.title(f"Correlación entre variables numéricas y {target_column}")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="distance", y=target_column, data=df)
plt.title("Distancia vs Retraso")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="weather_condition", y="departure_delay", data=df)
plt.title("Retrasos por Condición Climática")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="day_of_week", y="departure_delay", data=df)
plt.title("Retraso promedio por día de la semana")
plt.show()