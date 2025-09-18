import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos ficticios: horas de estudio vs notas
X = np.array([[1], [2], [3], [4], [5]])  # horas de estudio
y = np.array([1.0, 2.0, 3.2, 3.8, 4.2])       # notas

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Hacer predicciones
X_nuevo = np.array([[6]])  # ejemplo: 6 horas de estudio
prediccion = modelo.predict(X_nuevo)

print(f"Si estudias 6 horas, la nota esperada es: {prediccion[0]:.2f}")

# Visualización
plt.scatter(X, y, color="blue", label="Datos reales")
plt.plot(X, modelo.predict(X), color="red", label="Línea aprendida")
plt.scatter(6, prediccion, color="green", marker="x", s=100, label="Predicción")
plt.xlabel("Horas de estudio")
plt.ylabel("Nota")
plt.title("Regresión Lineal con Scikit-learn")
plt.legend()
plt.show()
