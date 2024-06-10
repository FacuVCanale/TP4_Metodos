import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
n = 5
d = 100
np.random.seed(42)

# Generación de matriz A y vector b aleatorios
A = np.random.randn(n, d)
b = np.random.randn(n)

# Función de costo F
def F(x):
    return np.dot((A @ x - b).T, A @ x - b)

# Gradiente de F
def grad_F(x):
    return 2 * A.T @ (A @ x - b)

# Función de costo F2 con regularización L2
def F2(x, delta):
    return F(x) + delta * np.dot(x, x)

# Gradiente de F2
def grad_F2(x, delta):
    return grad_F(x) + 2 * delta * x

# Inicialización de variables
sigma_max = np.linalg.svd(A, compute_uv=False)[0]
delta2 = 10**(-2) * sigma_max
lambda_max = np.linalg.eigvals(A.T @ A).max()
s = 1 / lambda_max
x0 = np.random.randn(d)
iterations = 1000

# Descenso por gradiente sin regularización
x = x0.copy()
cost_history_F = []

for _ in range(iterations):
    x = x - s * grad_F(x)
    cost_history_F.append(F(x))

# Descenso por gradiente con regularización L2
x_reg = x0.copy()
cost_history_F2 = []

for _ in range(iterations):
    x_reg = x_reg - s * grad_F2(x_reg, delta2)
    cost_history_F2.append(F2(x_reg, delta2))

# Solución usando SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
x_svd = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
cost_svd = F(x_svd)

# Gráficas
plt.figure(figsize=(10, 6))
plt.plot(cost_history_F, label='Descenso por gradiente')
plt.plot(cost_history_F2, label='Descenso por gradiente con regularización L2')
plt.axhline(y=cost_svd, color='r', linestyle='--', label='Solución SVD')
#plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Evolución del costo a lo largo de las iteraciones')
plt.legend()
plt.grid(True)
plt.show()

# Resultados finales
print(f"Solución por descenso por gradiente: F(x) = {cost_history_F[-1]}")
print(f"Solución por descenso por gradiente con regularización L2: F2(x) = {cost_history_F2[-1]}")
print(f"Solución por SVD: F(x_svd) = {cost_svd}")
