import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("C:/Users/admin/Desktop/methods/data.csv")

dataset_size = data['Dataset size'].values.astype(float)
train_time = data['Train time'].values.astype(float)


#  ФУНКЦІЯ РОЗДІЛЕНИХ РІЗНИЦЬ

def divided_differences(x, y):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef


def newton_polynomial(x_val, x_data, coef):
    n = len(coef)
    result = coef[-1]
    for i in range(n-2, -1, -1):
        result = result * (x_val - x_data[i]) + coef[i]
    return result



a = min(dataset_size)
b = max(dataset_size)

n = 5  
h = (b - a) / (n - 1)

x_tab = np.array([a + i*h for i in range(n)])

coef_original = divided_differences(dataset_size, train_time)
y_tab = [newton_polynomial(x, dataset_size, coef_original) for x in x_tab]

with open("tabulated_nodes.txt", "w") as f:
    f.write("Dataset size,Train time\n")
    for x, y in zip(x_tab, y_tab):
        f.write(f"{x},{y}\n")

print("Табуляція записана у tabulated_nodes.txt")



tab_data = pd.read_csv("tabulated_nodes.txt")

x_data = tab_data['Dataset size'].values
y_data = tab_data['Train time'].values

coeffs = divided_differences(x_data, y_data)

# ОБЧИСЛЕННЯ ДОВІЛЬНОГО ЗНАЧЕННЯ (120000)


x_target = 120000
predicted_time = newton_polynomial(x_target, x_data, coeffs)
print(f"\nПрогноз часу для 120000: {predicted_time:.2f} сек")


# ТабУЛЯЦІЯ ФУНКЦІЇ, ПОЛІНОМА І ПОХИБКИ


x_plot = np.linspace(a, b, 500)
y_true = [newton_polynomial(x, dataset_size, coef_original) for x in x_plot]
y_interp = [newton_polynomial(x, x_data, coeffs) for x in x_plot]
error = np.abs(np.array(y_true) - np.array(y_interp))

plt.figure(figsize=(9,5))
plt.scatter(dataset_size, train_time, color='red', s=80, label='Експериментальні дані')
plt.plot(x_plot, y_interp, color='blue', linewidth=2, label='Поліном Ньютона')
plt.plot(x_plot, error, color='green', linestyle='--', label='Похибка')
plt.xlabel('Dataset size')
plt.ylabel('Train time (sec)')
plt.title('Інтерполяція та похибка')
plt.legend()
plt.grid(True)
plt.show()



node_counts = [5, 10, 20]
colors = ['blue', 'green', 'purple']

plt.figure(figsize=(10,6))
errors_dict = {}

for n_nodes, color in zip(node_counts, colors):
    h_nodes = (b - a) / (n_nodes - 1)
    x_nodes = np.array([a + i*h_nodes for i in range(n_nodes)])
    y_nodes = [newton_polynomial(x, dataset_size, coef_original) for x in x_nodes]

    coef_nodes = divided_differences(x_nodes, y_nodes)
    y_interp_nodes = [newton_polynomial(x, x_nodes, coef_nodes) for x in x_plot]

    error_nodes = np.abs(np.array(y_true) - np.array(y_interp_nodes))
    errors_dict[n_nodes] = error_nodes

    plt.plot(x_plot, y_interp_nodes, linewidth=2, color=color, label=f'{n_nodes} вузлів')

plt.scatter(dataset_size, train_time, color='red', s=80, label='Експериментальні дані')
plt.xlabel('Dataset size')
plt.ylabel('Train time (sec)')
plt.title('Порівняння поліномів (5, 10, 20 вузлів)')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(10,6))
for n_nodes, color in zip(node_counts, colors):
    plt.plot(x_plot, errors_dict[n_nodes], linewidth=2, color=color, label=f'Похибка ({n_nodes} вузлів)')

plt.xlabel('Dataset size')
plt.ylabel('Абсолютна похибка')
plt.title('Порівняння похибок інтерполяції')
plt.legend()
plt.grid(True)
plt.show()