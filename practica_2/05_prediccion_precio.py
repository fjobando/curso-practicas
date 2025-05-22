import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Datos
metros = [[50], [60], [70], [80], [90]]
precios = [150000, 180000, 210000, 240000, 270000]

# Modelo
modelo = LinearRegression()
modelo.fit(metros, precios)

# Predicción
precio_estimado = modelo.predict([[85]])
print("Precio estimado para 85 m2:", precio_estimado[0])

# Gráfica
plt.scatter(metros, precios, color='blue')
plt.plot(metros, modelo.predict(metros), color='red')
plt.xlabel("Metros cuadrados")
plt.ylabel("Precio")
plt.title("Regresión lineal: Precio vs Metros cuadrados")
plt.grid(True)
plt.show()
