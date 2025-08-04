import numpy as np

def dolittle_lu(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A, dtype=float)

    print("\n--- DOOLITTLE METHOD ---")

    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
        print(f"\nAfter computing U row {i}:")
        print("L:")
        print(L)
        print("U:")
        print(U)

        for j in range(i+1, n):
            if U[i][i] == 0:
                raise ZeroDivisionError("Zero pivot encountered in Dolittle method.")
            L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]
        print(f"\nAfter computing L column {i}:")
        print("L:")
        print(L)
        print("U:")
        print(U)

    return L, U

def crout_lu(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    U = np.eye(n)

    print("\n--- CROUT METHOD ---")

    for j in range(n):
        for i in range(j, n):
            L[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(j))
        print(f"\nAfter computing L column {j}:")
        print("L:")
        print(L)
        print("U:")
        print(U)

        for i in range(j+1, n):
            if L[j][j] == 0:
                raise ZeroDivisionError("Zero pivot encountered in Crout method.")
            U[i][j+1:] = U[i][j+1:]  # No-op to preserve shape
            U[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(j))) / L[j][j]
        print(f"\nAfter computing U row {j}:")
        print("L:")
        print(L)
        print("U:")
        print(U)

    return L, U

def main():
    A = np.array([
        [2.0, -1.0, 1.0],
        [3.0, 3.0, 9.0],
        [3.0, 3.0, 5.0]
    ])

    print("Input Matrix A:")
    print(A)

    method = input("Select LU factorization method (dolittle = 1/crout = 2): ").strip()

    if method == '1':
        L, U = dolittle_lu(A)
    elif method == '2':
        L, U = crout_lu(A)
    else:
        print("Invalid method selected.type 1 or 2 nigguh")
        return

    print("\n=== Final Result ===")
    print("L matrix:")
    print(L)
    print("U matrix:")
    print(U)

    print("\nReconstructed A (L @ U):")
    print(np.round(L @ U, decimals=4))

if __name__ == "__main__":
    main()
