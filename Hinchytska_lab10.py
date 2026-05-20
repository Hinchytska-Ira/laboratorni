import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x + y  

def y_exact(x):
    return np.exp(x) - x - 1


def adams_method(f, a, b, y0, h):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    y[1] = y[0] + h * f(x[0], y[0])

    for i in range(1, n):
        y[i + 1] = y[i] + h / 2 * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))

    return x, y


def adams_predictor_corrector(f, a, b, y0, h):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    y[1] = y[0] + h * f(x[0], y[0])

    y_pred = np.zeros(n)
    y_corr = np.zeros(n)

    for i in range(1, n):
        y_pred[i] = y[i] + h * f(x[i], y[i])
        y_corr[i] = y[i] + h/2 * (f(x[i], y[i]) + f(x[i+1], y_pred[i]))
        y[i+1] = y_corr[i]

    return x, y, y_pred, y_corr


def runge_kutta(f, a, b, y0, h):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(x[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

    return x, y


def runge_error(f, a, b, y0, h):
    x_h, y_h = runge_kutta(f, a, b, y0, h)
    x_h2, y_h2 = runge_kutta(f, a, b, y0, h/2)

    y_h2_match = y_h2[::2]

    phi_runge = (y_h2_match - y_h) / 15

    return x_h, phi_runge


def auto_step(f, a, b, y0, tol):
    h = 0.1
    while True:
        x, y = adams_method(f, a, b, y0, h)
        err = max(abs(y - y_exact(x)))
        if err < tol:
            break
        h /= 2
    return h, x, y


def auto_step_rk(f, a, b, y0, tol):
    h = 0.1
    while True:
        x, y = runge_kutta(f, a, b, y0, h)
        err = max(abs(y - y_exact(x)))
        if err < tol:
            break
        h /= 2
    return h, x, y


a, b = 0, 1
y0 = 0
h = 0.1


x, y = adams_method(f, a, b, y0, h)

plt.plot(x, y - y_exact(x), 'o-')
plt.title('Локальна похибка методу Адамса')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.show()


x, y, y_pred, y_corr = adams_predictor_corrector(f, a, b, y0, h)

plt.plot(x[1:], y_corr - y_pred, 'o-')
plt.title('Оцінка похибки методом Адамса')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.show()


h_opt, x_opt, y_opt = auto_step(f, a, b, y0, 1e-3)

plt.plot(x_opt, y_opt, 'o-')
plt.title('Автоматичний вибір кроку (Адамс)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


x, y = runge_kutta(f, a, b, y0, h)

plt.plot(x, y - y_exact(x), 'o-')
plt.title('Локальна похибка RK4')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.show()


x_runge, phi_runge = runge_error(f, a, b, y0, h)

plt.plot(x_runge, phi_runge, 'o-')
plt.title('Оцінка похибки методом Рунге')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.show()


h_opt, x_opt, y_opt = auto_step_rk(f, a, b, y0, 1e-4)

plt.plot(x_opt, y_opt, 'o-')
plt.title('Автоматичний вибір кроку (RK4)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("Оптимальний крок h =", h_opt)


hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
errors = []

for h_test in hs:
    x_tmp, y_tmp = runge_kutta(f, a, b, y0, h_test)
    errors.append(max(abs(y_tmp - y_exact(x_tmp))))

plt.plot(hs, errors, 'o-')
plt.title('Залежність похибки від кроку')
plt.xlabel('h')
plt.ylabel('Похибка')
plt.gca().invert_xaxis()
plt.grid()
plt.show()

# перевірка оптимальності кроку за Рунге
tol = 1e-4
h = 0.1

while True:
    x_h, y_h = runge_kutta(f, a, b, y0, h)
    x_h2, y_h2 = runge_kutta(f, a, b, y0, h/2)

    y_h2_match = y_h2[::2]

    runge_est = max(abs(y_h2_match - y_h)) / 15

    if runge_est < tol:
        break

    h /= 2

print("Оптимальний крок за Рунге:", h)