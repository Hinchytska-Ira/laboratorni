import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24
x = np.linspace(a, b, 1000)
y = f(x)

plt.figure(figsize=(10,6))
plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()


#  Точне значення інтегралу

I_exact, _ = quad(f, a, b)
print(f"Точне значення інтегралу: {I_exact:.6f}")

def simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    S = y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-2:2])
    return S * h / 3

n = 10
I_simpson = simpson(f, a, b, n)
print(f"Інтеграл методом Сімпсона (n={n}): {I_simpson:.6f}")


#  Залежність похибки від n

ns = np.arange(10, 102, 2)
errors = [abs(simpson(f, a, b, n) - I_exact) for n in ns]

plt.figure(figsize=(10,6))
plt.plot(ns, errors, 'o-')
plt.yscale('log')
plt.xlabel('Число розбиттів n')
plt.ylabel('Похибка інтегрування')
plt.title('Залежність похибки інтегрування методом Сімпсона від n')
plt.grid(True)
plt.show()

eps = 1e-12
n_opt = min([n for n, e in zip(ns, errors) if e < eps])
print(f"Мінімальне n для досягнення точності {eps}: {n_opt}")
print(f"Похибка при n={n_opt}: {errors[ns.tolist().index(n_opt)]:.6e}")


#  Похибка при конкретному n

n0 = 4
I_num = simpson(f, a, b, n0)
error = abs(I_num - I_exact)
print(f"Похибка чисельного інтегрування при n={n0}: {error:.6e}")


#  Метод Рунге-Ромберга


I1 = simpson(f, a, b, n0//2)
I2 = simpson(f, a, b, n0)
p = 4
I_RR = I2 + (I2 - I1)/15
RR_error = abs(I_RR - I_exact)
print(f"Уточнене значення інтегралу методом Рунге-Ромберга: {I_RR:.6f}")
print(f"Похибка по Рунге-Ромбергу: {RR_error:.6e}")


#  Метод Ейткена

n_values = [n0//4, n0//2, n0]  


I_values = [simpson(f, a, b, n) for n in n_values]
I_n0, I_half, I_quarter = I_values[0], I_values[1], I_values[2]


chiselnik = I_half**2 - (I_n0 * I_quarter)
znamennik = 2 * I_half - (I_n0 + I_quarter)
I_Eitken = chiselnik / znamennik

Eitken_error = abs(I_Eitken - I_exact)
print(f"Уточнене значення інтегралу методом Ейткена: {I_Eitken:.6f}")
print(f"Похибка по Ейткену: {Eitken_error:.6e}")


print("\nПорівняння похибок методів:")
print(f"Сімпсон (n={n0}): {error:.6e}")
print(f"Рунге-Ромберг: {RR_error:.6e}")
print(f"Ейткен: {Eitken_error:.6e}")


# 9. Адаптивний алгоритм (Сімпсон)


function_calls = 0

def f_counted(x):
    global function_calls
    function_calls += 1
    return f(x)

def adaptive_simpson_recursive(a, b, delta):
    h = b - a
    mid = (a + b) / 2
    
    # Перше наближення
    I1 = (h / 6) * (f_counted(a) + 4 * f_counted(mid) + f_counted(b))
    
    # Друге (точніше) наближення
    quarter1 = (a + mid) / 2
    quarter2 = (mid + b) / 2
    
    I2 = (h / 12) * (f_counted(a) + 4 * f_counted(quarter1) + f_counted(mid)) + \
         (h / 12) * (f_counted(mid) + 4 * f_counted(quarter2) + f_counted(b))
    
    # Умова точності
    if abs(I1 - I2) <= delta:
        return I2
    else:
        return adaptive_simpson_recursive(a, mid, delta / 2) + \
               adaptive_simpson_recursive(mid, b, delta / 2)


function_calls = 0
eps = 1e-12


I_adaptive = adaptive_simpson_recursive(a, b, eps)

print(f"Адаптивний метод Сімпсона: {I_adaptive:.6f}")
print(f"Похибка: {abs(I_adaptive - I_exact):.6e}")
print(f"Кількість викликів f(x): {function_calls}")