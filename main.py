import numpy as np
import matplotlib.pyplot as plt

n = 100
d = 5
np.random.seed(0)

A = np.random.randn(n, d)
b = np.random.randn(n)

def F(x): return np.dot((A @ x - b).T, A @ x - b)

def grad_F(x): return 2 * A.T @ (A @ x - b)

def F2(x, delta): return F(x) + delta * np.dot(x, x)

def grad_F2(x, delta): return grad_F(x) + 2 * delta * x

sigma_max = np.linalg.svd(A, compute_uv=False)[0]
delta2 = 10**(-2) * sigma_max
lambda_max = np.linalg.eigvals(2 * A.T @ A).max()
s = 1 / lambda_max
x0 = np.random.randn(d)
iterations = 1000

x = x0.copy()
cost_history_F = []
norm_history_F = []
x_reg = x0.copy()
cost_history_F2 = []
norm_history_F2 = []

for _ in range(iterations):
    x = x - s * grad_F(x)
    cost_history_F.append(F(x))
    norm_history_F.append(np.linalg.norm(x))

    x_reg = x_reg - s * grad_F2(x_reg, delta2)
    cost_history_F2.append(F2(x_reg, delta2))
    norm_history_F2.append(np.linalg.norm(x_reg))

U, S, Vt = np.linalg.svd(A, full_matrices=False)
x_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
cost_svd = F(x_svd)

norm_svd = np.linalg.norm(x_svd)

plt.figure(figsize=(10, 6))
plt.plot(cost_history_F, label='Descenso por gradiente')
plt.plot(cost_history_F2, label='Descenso por gradiente con regularización L2')
plt.axhline(y=cost_svd, color='r', linestyle='--', label='Solución SVD')
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Evolución del costo a lo largo de las iteraciones')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(norm_history_F, label='Descenso por gradiente')
plt.plot(norm_history_F2, label='Descenso por gradiente con regularización L2')
plt.axhline(y=norm_svd, color='r', linestyle='--', label='Solución SVD')
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Iteraciones')
plt.ylabel('Norma de x')
plt.title('Evolución de la norma de x a lo largo de las iteraciones')
plt.legend()
plt.grid(True)
plt.show()

# Resultados finales
print(f"Solución por descenso por gradiente: F(x) = {cost_history_F[-1]}")
print(f"Solución por descenso por gradiente con regularización L2: F2(x) = {cost_history_F2[-1]}")
print(f"Solución por SVD: F(x_svd) = {cost_svd}")

# # Generación de datos para el gráfico de Ridge Regression usando A y b
# # Parámetros de regularización
# lambdas = np.logspace(0, 3, 100)  # Ajuste de 10^0 a 10^3

# # Ajuste del modelo Ridge para diferentes valores de lambda
# coefs = []
# for l in lambdas:
#     ridge = Ridge(alpha=l, fit_intercept=False)
#     ridge.fit(A, b)
#     coefs.append(ridge.coef_)

# coefs = np.array(coefs)

# # Gráfico del camino de los coeficientes de Ridge Regression
# plt.figure(figsize=(10, 6))
# for i in range(d):
#     plt.plot(lambdas, coefs[:, i], label=f'Feature {i+1}')

# plt.xscale('log')
# plt.xlabel(r'$\lambda$')
# plt.ylabel('Coeficientes de los Features')
# plt.title('Ridge Regression Path')
# plt.legend()
# plt.grid(True)
# plt.show()


# como un plus, agregar cambiando restricción
# plotear valores de X vector en cant de iteraciones
# plotear norma