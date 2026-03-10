import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("temperature.csv")
data.columns = data.columns.str.strip()  # пробіли в назвах


def tabulate_data(x_start, x_end, n_points, y_values):

    x_nodes = []
    h = (x_end - x_start) / (n_points - 1)

    for i in range(n_points):
        x = x_start + i * h
        x_nodes.append(x)

    x_nodes = np.array(x_nodes)

    return x_nodes, y_values

y_data = data["Temp"].values

x_nodes, y_nodes = tabulate_data(1, 24, 24, y_data)

def form_matrix(x, m):
    A = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            A[i,j] = np.sum(x**(i+j))
    return A

def form_vector(x, y, m):
    b = np.zeros(m+1)
    for i in range(m+1):
        b[i] = np.sum(y * x**i)
    return b


#  Гаус 

def gauss_solve(A, b):
    n = len(b)
    for k in range(n):
        max_row = np.argmax(abs(A[k:,k])) + k
        A[[k,max_row]] = A[[max_row,k]]
        b[[k,max_row]] = b[[max_row,k]]
        for i in range(k+1,n):
            factor = A[i,k] / A[k,k]
            A[i,k:] = A[i,k:] - factor*A[k,k:]
            b[i] = b[i] - factor*b[k]
    x_sol = np.zeros(n)
    for i in range(n-1,-1,-1):
        x_sol[i] = (b[i] - np.sum(A[i,i+1:] * x_sol[i+1:])) / A[i,i]
    return x_sol

def polynomial(x, coef):
    y_poly = np.zeros_like(x,dtype=float)
    for i in range(len(coef)):
        y_poly += coef[i]*x**i
    return y_poly

def variance(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


#  оптимальний степінь полінома

max_degree = 4
variances = []
all_coefs = []

for m in range(1, max_degree+1):
    A = form_matrix(x_nodes,m)
    b_vec = form_vector(x_nodes,y_nodes,m)
    coef = gauss_solve(A.copy(),b_vec.copy())
    y_approx = polynomial(x_nodes, coef)
    var = variance(y_nodes, y_approx)
    variances.append(var)
    all_coefs.append(coef)
    print("Степінь:", m, " Дисперсія:", var)

optimal_m = np.argmin(variances) + 1
coef_opt = all_coefs[optimal_m-1]
print("\nОптимальний степінь:", optimal_m)
print("Коефіцієнти полінома:", coef_opt)


x_future = np.array([25,26,27])
y_future = polynomial(x_future, coef_opt)
print("\nПрогноз температур:")
for i in range(3):
    print("Місяць", x_future[i], "=", y_future[i])


error_table = []

n_intervals = len(x_nodes) - 1
h1 = (x_nodes[-1] - x_nodes[0]) / (20 * n_intervals)  
x_fine = np.arange(x_nodes[0], x_nodes[-1] + h1, h1)

for x in x_fine:
    phi = polynomial(np.array([x]), coef_opt)[0]
    real = np.interp(x, x_nodes, y_nodes)  # лінійна інтерполяція для проміжних точок
    error = abs(real - phi)
    error_table.append((x, phi, error))
    


print("\nТабуляція похибки (перші 5 значень):")
print(f"{'x':>10} | {'phi(x)':>10} | {'Error':>10}")
print("-"*35)
for x, phi, e in error_table[:5]:
    print(f"{x:10.4f} | {phi:10.4f} | {e:10.4f}")



plt.figure(figsize=(10,6))
plt.scatter(x_nodes, y_nodes, color="red", label="Дані")
plt.plot(x_nodes, polynomial(x_nodes, coef_opt), label="Апроксимація")
plt.xlabel("Місяць")
plt.ylabel("Температура")
plt.title("Апроксимація температури")
plt.legend()
plt.grid(True)
plt.show()


plt.figure()
plt.plot(range(1,max_degree+1),variances,'o-')
plt.xlabel("Степінь полінома")
plt.ylabel("Дисперсія")
plt.title("Залежність дисперсії від степеня")
plt.grid(True)
plt.show()


x_err, err_vals = zip(*[(x, e) for x, _, e in error_table])
plt.figure()
plt.plot(x_err, err_vals)
plt.xlabel("Місяць")
plt.ylabel("Похибка")
plt.title("Похибка апроксимації")
plt.grid(True)
plt.show()