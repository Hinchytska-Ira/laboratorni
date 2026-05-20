import random
import math
import matplotlib.pyplot as plt

N = 100

def generate_matrix():
    A = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(random.randint(1, 9))
        A.append(row)
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


def generate_B(A,x):
    B = []
    for i in range(N):
        s = 0
        Ai = A[i]
        for j in range(N):
            s += Ai[j] * x[j]
        B.append(s)
    return B


def lu(A):
    L = [[0]*N for _ in range(N)]
    U = [[0]*N for _ in range(N)]

    for i in range(N):
        for j in range(i, N):
            s = 0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - s

        for j in range(i, N):
            if i == j:
                L[i][i] = 1
            else:
                s = 0
                for k in range(i):
                    s += L[j][k] * U[k][i]

                #  від ділення на 0
                if abs(U[i][i]) < 1e-12:
                    U[i][i] = 1e-12

                L[j][i] = (A[j][i] - s) / U[i][i]

    return L, U


def write_LU(L, U):
    with open("LU.txt", "w") as f:
        f.write("L:\n")
        for row in L:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\nU:\n")
        for row in U:
            f.write(" ".join(map(str, row)) + "\n")


def solve(L, U, B):
    y = [0]*N

    for i in range(N):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = B[i] - s

    x = [0]*N

    for i in range(N-1, -1, -1):
        s = 0
        for j in range(i+1, N):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]

    return x


def get_error(A, x, B):
    max_err = 0
    for i in range(N):
        s = 0
        Ai = A[i]
        for j in range(N):
            s += Ai[j] * x[j]
        e = abs(s - B[i])
        if e > max_err:
            max_err = e
    return max_err


def refine(A, L, U, B, x):
    errors = []
    eps = 1e-11
    max_iter = 100
    it_count = 0   

    for it in range(max_iter):

        r = []
        for i in range(N):
            s = 0
            Ai = A[i]
            for j in range(N):
                s += Ai[j] * x[j]
            r.append(B[i] - s)

        delta = solve(L, U, r)

        norm = 0
        for d in delta:
            norm += d*d
        norm = math.sqrt(norm)

        errors.append(norm)

        for i in range(N):
            x[i] += delta[i]

        it_count += 1  
        if norm < eps:
            break

    return x, errors, it_count



A = generate_matrix()

x_true = [2.5] * N

B = generate_B(A,x_true)


write_matrix(A)
write_vector(B)


A = read_matrix()
B = read_vector()

L, U = lu(A)

write_LU(L, U)

x = solve(L, U, B)

print("Початкова похибка:", get_error(A, x, B))


x, errors, it_count = refine(A, L, U, B, x)

print("Кількість ітерацій:", it_count)

print("Похибка після уточнення:", get_error(A, x, B))


if len(errors) > 1:
    plt.plot(errors, marker='o')
    plt.yscale("log")
    plt.xlabel("Ітерація")
    plt.ylabel("Норма поправки")
    plt.title("Збіжність уточнення")
    plt.grid()
    plt.show()
else:
    print("Замало ітерацій для графіка")