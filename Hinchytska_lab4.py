import numpy as np
import matplotlib.pyplot as plt

# Функція та її похідна
def f(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def df(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

x0 = 1.0


def derivative(h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

true_value = df(x0)


h_values = np.logspace(-6, -1, 100) 
errors = []

for h in h_values:
    approx = derivative(h)
    error = abs(approx - true_value)
    errors.append(error)


min_index = np.argmin(errors)
h_opt = h_values[min_index]
R0 = errors[min_index]

print("Оптимальний крок h0 =", h_opt)
print("Мінімальна похибка R0 =", R0)

plt.figure()
plt.loglog(h_values, errors)
plt.xlabel("h")
plt.ylabel("Похибка")
plt.title("Залежність похибки від кроку")
plt.grid()
plt.show()


h = 0.001  


y_h = (f(x0 + h) - f(x0 - h)) / (2 * h)
y_2h = (f(x0 + 2*h) - f(x0 - 2*h)) / (4 * h)
R1 = abs(y_h - true_value)

print("\nПохідна при h:", y_h)
print("Похідна при 2h:", y_2h)
print("Похибка R1:", R1)

# Рунге
y_rr = y_h + (y_h - y_2h)/3
R2 = abs(y_rr - true_value)

print("\nРунге-Ромберг:", y_rr)
print("Похибка R2:", R2)

#  Ейткен 
y_4h = (f(x0 + 4*h) - f(x0 - 4*h)) / (8 * h)

denominator = y_4h - 2*y_2h + y_h
if abs(denominator) < 1e-12:
    y_e = y_h  
else:
    y_e = y_h - ((y_2h - y_h)**2) / denominator


if abs(y_2h - y_h) < 1e-12:
    p = np.nan
else:
    p = np.log(abs((y_4h - y_2h)/(y_2h - y_h))) / np.log(2)

R3 = abs(y_e - true_value)

print("\nЕйткен:", y_e)
print("Порядок точності p:", p)
print("Похибка R3:", R3)


h_vals_small = np.logspace(-4, -2, 50)  
errors_basic = []
errors_rr = []
errors_aitken = []

for h in h_vals_small:
    y_h = (f(x0 + h) - f(x0 - h)) / (2 * h)
    y_2h = (f(x0 + 2*h) - f(x0 - 2*h)) / (4 * h)
    y_4h = (f(x0 + 4*h) - f(x0 - 4*h)) / (8 * h)

    # Рунге-Ромберг
    y_rr = y_h + (y_h - y_2h)/3

    # Ейткен
    denom = y_4h - 2*y_2h + y_h
    if abs(denom) < 1e-12:
        y_e = y_h
    else:
        y_e = y_h - ((y_2h - y_h)**2) / denom

    errors_basic.append(abs(y_h - true_value))
    errors_rr.append(abs(y_rr - true_value))
    errors_aitken.append(abs(y_e - true_value))

plt.figure()
plt.loglog(h_vals_small, errors_basic, label="Без уточнення")
plt.loglog(h_vals_small, errors_rr, label="Рунге-Ромберг")
plt.loglog(h_vals_small, errors_aitken, label="Ейткен")
plt.legend()
plt.xlabel("h")
plt.ylabel("Похибка")
plt.title("Порівняння методів чисельного диференціювання")
plt.grid()
plt.show()