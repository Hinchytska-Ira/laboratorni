import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return np.cos(x) - x/3

def dF(x):
    return -np.sin(x) - 1/3

def d2F(x):
    return -np.cos(x)

a, b = -10, 10
h = 0.1
table = []

with open("C:/Users/admin/Desktop/methods/tabulation1.txt", "w") as f:
    x = a
    while x <= b + h/2:
        y = F(x)
        table.append((x, y))
        f.write(f"{x:.2f}\t{y:.6f}\n")
        x += h

# Графік 
xs = [p[0] for p in table]
ys = [p[1] for p in table]

plt.plot(xs, ys)
plt.axhline(0, color='black')
plt.grid()
plt.title("F(x) = cos(x) - x/3")
plt.show()

#  Пошук коренів 
roots = []
for i in range(len(table)-1):
    if table[i][1]*table[i+1][1] < 0:
        roots.append((table[i][0] + table[i+1][0]) / 2)

if len(roots) < 2:
    print("Недостатньо коренів")
    exit()

# вибір різної поведінки
x_start_1 = None
x_start_2 = None

for r in roots:
    if dF(r) > 0 and x_start_1 is None:
        x_start_1 = r
    if dF(r) < 0 and x_start_2 is None:
        x_start_2 = r

# fallback
if x_start_1 is None or x_start_2 is None:
    x_start_1 = roots[0]
    x_start_2 = roots[-1]

print("Початкові точки:", x_start_1, x_start_2)

#  Методи 

def simple_iteration(x0, eps=1e-10):
    phi = lambda x: x - 0.5 * F(x)
    x = x0
    for it in range(1000):
        x1 = phi(x)
        if abs(F(x1)) < eps and abs(x1 - x) < eps:
            return x1, it
        x = x1
    return x, it

def newton(x0, eps=1e-10):
    x = x0
    for it in range(1000):
        df = dF(x)
        if df == 0:
            break
        x1 = x - F(x)/df
        if abs(F(x1)) < eps and abs(x1 - x) < eps:
            return x1, it
        x = x1
    return x, it

def chebyshev(x0, eps=1e-10):
    x = x0
    for it in range(1000):
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)
        if dfx == 0:
            break
        x1 = x - fx/dfx - (d2fx * fx**2)/(2 * dfx**3)
        if abs(F(x1)) < eps and abs(x1 - x) < eps:
            return x1, it
        x = x1
    return x, it

def secant(x0, x1, eps=1e-10):
    for it in range(1000):
        f0, f1 = F(x0), F(x1)
        if f1 - f0 == 0:
            break
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        if abs(F(x2)) < eps and abs(x2 - x1) < eps:
            return x2, it
        x0, x1 = x1, x2
    return x1, it

def muller(x0, x1, x2, eps=1e-10):
    for it in range(1000):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        h1 = x1 - x0
        h2 = x2 - x1
        if h1 == 0 or h2 == 0:
            break
        d1 = (f1 - f0)/h1
        d2 = (f2 - f1)/h2
        d = (d2 - d1)/(h1 + h2)
        b = d2 + h2*d
        D = np.sqrt(b*b - 4*f2*d)
        E = b + D if abs(b-D) < abs(b+D) else b - D
        if E == 0:
            break
        x3 = x2 - 2*f2/E
        if abs(F(x3)) < eps and abs(x3 - x2) < eps:
            return x3, it
        x0, x1, x2 = x1, x2, x3
    return x2, it

def inverse_interp(x0, x1, x2, eps=1e-10):
    for it in range(1000):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        d0 = (f0-f1)*(f0-f2)
        d1 = (f1-f0)*(f1-f2)
        d2 = (f2-f0)*(f2-f1)
        if d0 == 0 or d1 == 0 or d2 == 0:
            break
        x3 = (x0*f1*f2/d0 + x1*f0*f2/d1 + x2*f0*f1/d2)
        if abs(F(x3)) < eps and abs(x3 - x2) < eps:
            return x3, it
        x0, x1, x2 = x1, x2, x3
    return x2, it


methods = [
    ("Ньютон", lambda x: newton(x)),
    ("Проста ітерація", lambda x: simple_iteration(x)),
    ("Чебишев", lambda x: chebyshev(x)),
    ("хорди", lambda x: secant(x, x+0.1)),
    ("Мюллера", lambda x: muller(x, x+0.1, x+0.2)),
    ("Зворотна інтерполяція", lambda x: inverse_interp(x, x+0.1, x+0.2)),
]

for i, start in enumerate([x_start_1, x_start_2], 1):
    print(f"\n=== Корінь {i} ===")
    for name, method in methods:
        root, it = method(start)
        print(f"{name}: {root:.10f}, ітерацій: {it}")

#  Поліном 
coeffs = [1, -2, 1, -2]

with open("poly.txt", "w") as f:
    f.write(" ".join(map(str, coeffs)))

def read_poly(file):
    with open(file) as f:
        return list(map(float, f.read().split()))

def horner(poly, x):
    val = poly[0]
    der = 0
    for i in range(1, len(poly)):
        der = der*x + val
        val = val*x + poly[i]
    return val, der

def newton_poly(poly, x0, eps=1e-10):
    x = x0
    for it in range(1000):
        val, der = horner(poly, x)
        if der == 0:
            break
        x1 = x - val/der
        if abs(val) < eps and abs(x1-x) < eps:
            return x1, it
        x = x1
    return x, it

poly = read_poly("poly.txt")
root, it = newton_poly(poly, 3)
print("\nДійсний корінь:", root, "ітерацій:", it)

# Метод Ліна 
def lina(poly, eps=1e-10):
    # кілька спроб з різними початковими значеннями
    starts = [(0,1), (1,1), (-1,1), (0.5,0.5), (1,-1)]

    for p_init, q_init in starts:
        p, q = p_init, q_init

        for it in range(1000):
            b = [poly[0]]
            for i in range(1, len(poly)):
                b.append(poly[i] + p*b[i-1] + (q*b[i-2] if i > 1 else 0))

            c = [b[0]]
            for i in range(1, len(b)-1):
                c.append(b[i] + p*c[i-1] + (q*c[i-2] if i > 1 else 0))

            # 🔴 ЗАХИСТ ВІД ДІЛЕННЯ НА 0
            if abs(c[-2]) < 1e-12:
                break

            dp = -b[-2]/c[-2]
            dq = -b[-1]/c[-2]

            p += dp
            q += dq

            if abs(dp) < eps and abs(dq) < eps:
                D = p*p - 4*q
                x1 = (-p + np.sqrt(D))/2
                x2 = (-p - np.sqrt(D))/2
                return x1, x2, it

    print("Метод Ліна не зійшовся")
    return None, None, 0

c1, c2, it = lina(poly)
print("Комплексні корені:", c1, c2, "ітерацій:", it)
# ГРАФІК ПОЛІНОМА
x_vals = np.linspace(-3, 3, 200)
y_vals = []

for x in x_vals:
    val = 0
    for i, c in enumerate(coeffs):
        val += c * x**(len(coeffs)-1-i)
    y_vals.append(val)

import matplotlib.pyplot as plt
plt.plot(x_vals, y_vals)
plt.axhline(0, color='black')
plt.title("Поліном 3-го степеня")
plt.grid()
plt.show()