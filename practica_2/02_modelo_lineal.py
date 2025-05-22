from sklearn.linear_model import LinearRegression
import numpy as np

# Datos de entrenamiento
X = np.array([[1], [2], [3], [4], [5]])  # Horas estudiadas
y = np.array([2, 4, 6, 8, 10])           # Calificaciones

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Nuevos datos para predecir
nuevos_datos = np.array([[6], [7]])
predicciones = modelo.predict(nuevos_datos)

print("Predicciones para 6 y 7 horas:", predicciones)
