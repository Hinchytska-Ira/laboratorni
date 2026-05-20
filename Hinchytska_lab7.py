import random
import matplotlib.pyplot as plt

N = 100
eps = 1e-8


def generate_matrix():
    A = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(random.randint(1, 9))
        A.append(row)

    # діагональне переважання
    for i in range(N):
        s = 0
        for j in range(N):
            if i != j:
                s += abs(A[i][j])
        A[i][i] = s + 2

    return A


def write_matrix(A):
    with open("A.txt", "w") as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")


def write_vector(B):
    with open("B.txt", "w") as f:
        for v in B:
            f.write(str(v) + "\n")


def read_matrix():
    A = []
    with open("A.txt", "r") as f:
        for line in f:
            A.append(list(map(float, line.split())))
    return A


def read_vector():
    B = []
    with open("B.txt", "r") as f:
        for line in f:
            B.append(float(line))
    return B


def generate_B(A, x):
    B = []
    for i in range(N):
        s = 0
        for j in range(N):
            s += A[i][j] * x[j]
        B.append(s)
    return B



def norm_vec(v):
    m = 0
    for i in v:
        if abs(i) > m:
            m = abs(i)
    return m


def norm_matrix(A):
    max_sum = 0
    for i in range(N):
        s = 0
        for j in range(N):
            s += abs(A[i][j])
        if s > max_sum:
            max_sum = s
    return max_sum


#  C = E - τA перевірка збіжності
def make_C(A, tau):
    C = []
    for i in range(N):
        row = []
        for j in range(N):
            if i == j:
                row.append(1 - tau * A[i][j])
            else:
                row.append(-tau * A[i][j])
        C.append(row)
    return C



x0 = []
for i in range(N):
    x0.append(1)



def simple_iter(A, B):
    x = x0[:]
    k = 0
    errors = []

    normA = norm_matrix(A)
    tau = 0.5 / normA

    while True:
        # Ax
        Ax = []
        for i in range(N):
            s = 0
            for j in range(N):
                s += A[i][j] * x[j]
            Ax.append(s)

        # формула
        x_new = []
        for i in range(N):
            x_new.append(x[i] - tau * (Ax[i] - B[i]))

        diff = []
        for i in range(N):
            diff.append(x_new[i] - x[i])

        err = norm_vec(diff)
        errors.append(err)

        if err < eps:
            break

        x = x_new[:]
        k += 1

    return x, k, errors


def jacobi(A, B):
    x = x0[:]   #копія списку
    k = 0
    errors = []

    while True:
        x_new = [0] * N

        for i in range(N):
            s = 0
            for j in range(N):
                if j != i:
                    s += A[i][j] * x[j]
            x_new[i] = (B[i] - s) / A[i][i]

        diff = []
        for i in range(N):
            diff.append(x_new[i] - x[i])

        err = norm_vec(diff)
        errors.append(err)

        if err < eps:
            break

        x = x_new[:]
        k += 1

    return x, k, errors


def zeidel(A, B):
    x = x0[:]
    k = 0
    errors = []

    while True:
        x_old = x[:]

        for i in range(N):
            s1 = 0
            s2 = 0

            for j in range(i):
                s1 += A[i][j] * x[j]

            for j in range(i + 1, N):
                s2 += A[i][j] * x_old[j]

            x[i] = (B[i] - s1 - s2) / A[i][i]

        diff = []
        for i in range(N):
            diff.append(x[i] - x_old[i])

        err = norm_vec(diff)
        errors.append(err)

        if err < eps:
            break

        k += 1

    return x, k, errors



A = generate_matrix()

x_true = []
for i in range(N):
    x_true.append(2.5)

B = generate_B(A, x_true)

write_matrix(A)
write_vector(B)

A = read_matrix()
B = read_vector()


normA = norm_matrix(A)
tau = 0.5 / normA
C = make_C(A, tau)
print("A =", normA)
print("C =", norm_matrix(C))

# методи
x_s, k1, e1 = simple_iter(A, B)
x_j, k2, e2 = jacobi(A, B)
x_z, k3, e3 = zeidel(A, B)

print("Проста ітерація:", k1)
print("Якобі:", k2)
print("Зейдель:", k3)


plt.plot(e1, label="Проста ітерація")
plt.plot(e2, label="Якобі")
plt.plot(e3, label="Зейдель")

plt.yscale("log")
plt.xlabel("Ітерації")
plt.ylabel("Похибка")
plt.title("Порівняння методів")
plt.legend()

plt.show()