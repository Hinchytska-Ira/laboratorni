import numpy as np
import matplotlib.pyplot as plt


#  СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ (m = 2)

# x^2 - y = 0
# x - 1 = 0

def f(x, y):
     return (2 - x)**2 + 100 * (y - x**2)**2


x_line = np.linspace(-2, 2, 400)

y_parabola = x_line**2

plt.figure(figsize=(7, 5))

plt.plot(x_line, y_parabola, label="y = x^2")
plt.axvline(2, color='black', label="x = 2")

plt.scatter(1, 1, c='red', s=80, label="Розв’язок (1,1)")

plt.title("Графіки рівнянь системи")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()

plt.show()


x0, y0 = -1.2, 0.0
step0 = 0.5



#  МЕТОД ХУКА-ДЖИВСА

def hooke_jeeves(x0, y0, step, alpha=0.5, eps=1e-6, max_iter=1000):
    x, y = x0, y0
    traj = [(x, y)]

    def explore(x, y, step):
        best_x, best_y = x, y

        if f(x + step, y) < f(x, y):
            best_x = x + step
        elif f(x - step, y) < f(x, y):
            best_x = x - step

        if f(best_x, y + step) < f(best_x, y):
            best_y = y + step
        elif f(best_x, y - step) < f(best_x, y):
            best_y = y - step

        return best_x, best_y

    for _ in range(max_iter):
        x_new, y_new = explore(x, y, step)

        if f(x_new, y_new) < f(x, y):
            x_pattern = x_new + (x_new - x)
            y_pattern = y_new + (y_new - y)

            x2, y2 = explore(x_pattern, y_pattern, step)

            if f(x2, y2) < f(x_new, y_new):
                x, y = x2, y2
            else:
                x, y = x_new, y_new
        else:
            step *= alpha

        traj.append((x, y))

        if step < eps:
            break

    return x, y, traj


x_min, y_min, traj = hooke_jeeves(x0, y0, step0)

print("Мінімум:")
print("x =", x_min)
print("y =", y_min)
print("f =", f(x_min, y_min))
print("Кроків:", len(traj))


with open("trajectory.txt", "w") as file:
    for p in traj:
        file.write(f"{p}\n")


x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(7, 5))

plt.contour(X, Y, Z, 10, colors='black')

traj = np.array(traj)

plt.plot(traj[:, 0], traj[:, 1], 'r-', label="Хук-Дживс")
plt.scatter(x_min, y_min, c='blue', s=80, label="Мінімум (1,1)")

plt.title("Метод Хука-Дживса")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()

plt.show()