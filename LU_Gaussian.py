import numpy as np

def LU_gaussian(A):
    A = A.astype(float)
    n = A.shape[0]
    L = np.eye(n) 
    U = A.copy()

    print("Initial Matrix A:")
    print(A)

    print("="*50)

    for k in range(n-1):
        print(f"\n=== Eliminating column {k} ===")
        for i in range(k+1, n):
            if U[k, k] == 0:
                raise ZeroDivisionError("Zero pivot encountered!")
            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            U[i, k:] -= multiplier * U[k, k:]

            print(f"\nRow{i} updated using row {k}")
            print(f"Multiplier: L[{i}{k}] = {multiplier:.4f}")
            print("Updated Row in U:", U[i])

        print("\nCurrent L matrix:")
        print(L)
        print("\nCurrent U matrix:")
        print(U)
    print("\n=== Final LU Factorization ===")
    print("L:", L)
    print("U:", U)
