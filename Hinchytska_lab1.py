import requests
import numpy as np
import matplotlib.pyplot as plt


url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
results = response.json()["results"]

n = len(results)

lat = [p["latitude"] for p in results]
lon = [p["longitude"] for p in results]
elev = np.array([p["elevation"] for p in results])


with open("C:/Users/admin/Desktop/tabulation.txt", "w", encoding="utf-8") as f:
    f.write("№ | Latitude | Longitude | Elevation\n")
    for i in range(n):
        f.write(f"{i:2d} | {lat[i]:.6f} | {lon[i]:.6f} | {elev[i]:.2f}\n")


# Кумулятивна відстань
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))



dist = [0]
for i in range(1, n):
    d = haversine(lat[i-1], lon[i-1], lat[i], lon[i])
    dist.append(dist[-1] + d)

x = np.array(dist)     # кумулятивна відстань 
y = elev               

print("Загальна довжина маршруту (м):", x[-1])


# Метод прогонки
def method_progonka(x, y):
    n = len(x)
    h = np.diff(x)

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    B[0] = B[-1] = 1

    for i in range(1, n-1):
        A[i] = h[i-1]
        B[i] = 2*(h[i-1] + h[i])
        C[i] = h[i]
        D[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    for i in range(1, n):
        m = A[i]/B[i-1]
        B[i] -= m*C[i-1]
        D[i] -= m*D[i-1]

    M = np.zeros(n)
    M[-1] = D[-1]/B[-1]

  
    for i in range(n-2, -1, -1):
        M[i] = (D[i] - C[i]*M[i+1]) / B[i]

    return M


M_full = method_progonka(x, y)


# Сплайн
def spline_eval(x, y, M, xi):
    n = len(x)
    h = np.diff(x)
    for i in range(n-1):
        if x[i] <= xi <= x[i+1]:
            dx = xi - x[i]
            return (y[i]
                    + ((y[i+1]-y[i])/h[i] - h[i]*(2*M[i]+M[i+1])/6)*dx
                    + (M[i]/2)*dx**2
                    + (M[i+1]-M[i])/(6*h[i])*dx**3)
    return None


xx = np.linspace(x[0], x[-1], 500)
yy_full = np.array([spline_eval(x, y, M_full, xi) for xi in xx])


node_sets = [10, 15, 20]


plt.figure()
plt.plot(xx, yy_full, label="Повний сплайн", linewidth=3, color="black")

errors_dict = {}

for nodes in node_sets:

    idx = np.linspace(0, n-1, nodes, dtype=int)
    x_sub = x[idx]
    y_sub = y[idx]

    M_sub = method_progonka(x_sub, y_sub)
    yy_sub = np.array([spline_eval(x_sub, y_sub, M_sub, xi) for xi in xx])
    error = np.abs(yy_full - yy_sub)

    errors_dict[nodes] = error

    print(f"\n=== {nodes} вузлів ===")
    print("Максимальна похибка:", np.max(error))
    print("Середня похибка:", np.mean(error))

    plt.plot(xx, yy_sub, label=f"{nodes} вузлів")
    plt.scatter(x_sub, y_sub, s=40)

plt.title("Порівняння сплайнів")
plt.legend()
plt.grid()
plt.show()



plt.figure()

for nodes in node_sets:
    plt.plot(xx, errors_dict[nodes], label=f"{nodes} вузлів")

plt.title("Похибки для різної кількості вузлів")
plt.legend()
plt.grid()
plt.show()



total_ascent = sum(max(y[i]-y[i-1], 0) for i in range(1, n))
total_descent = sum(max(y[i-1]-y[i], 0) for i in range(1, n))

print("Сумарний набір висоти (м):", total_ascent)
print("Сумарний спуск (м):", total_descent)

grad_full = np.gradient(yy_full, xx) * 100
print("Максимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))


mass = 80
g = 9.81
energy = mass * g * total_ascent

print("Механічна робота (кДж):", energy/1000)
print("Енергія (ккал):", energy/4184)