import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Datos para graficar la línea
X_linea = np.linspace(1, 6, 100).reshape(-1, 1)
y_pred = modelo.predict(X_linea)

# Graficar
plt.scatter(X, y, color='blue')
plt.plot(X_linea, y_pred, color='red')
plt.xlabel("Horas estudiadas")
plt.ylabel("Calificación")
plt.title("Modelo de Regresión Lineal")
plt.grid(True)
plt.show()
