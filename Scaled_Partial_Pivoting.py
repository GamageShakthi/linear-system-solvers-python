import numpy as np

# Scaled Partial Pivoting with tracing
def scaled_partial_pivoting(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    print("Initial Matrix A:")
    print(A)
    print("Initial Vector b:")
    print(b)
    print("="*50)

    # Scaling factors
    scale = np.max(np.abs(A), axis=1)

    for k in range(n - 1):
        # Select pivot row
        ratios = np.abs(A[k:n, k]) / scale[k:n]
        max_index = np.argmax(ratios) + k

        print(f"\nStep {k + 1}:")
        print(f"Pivot column = {k}")
        print(f"Scaling factors = {scale[k:n]}")
        print(f"Ratios = {ratios}")
        print(f"Selected pivot row: {max_index}")

        if A[max_index, k] == 0:
            raise ValueError("Zero pivot encountered!")

        # Swap rows
        if max_index != k:
            print(f"Swapping row {k} with row {max_index}")
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]
            scale[[k, max_index]] = scale[[max_index, k]]

        # Elimination
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
            print(f"Eliminating row {i}, factor = {factor:.4f}")
            print("Matrix A now:")
            print(A)
            print("Vector b now:")
            print(b)

    print("\n=== Upper Triangular Matrix (after elimination) ===")
    print(A)
    print("Modified b:")
    print(b)

    print("\n=== Final System of Equations ===")
    for i in range(n):
        terms = " + ".join([f"{A[i,j]:.3f}*x{j+1}" for j in range(n)])
        print(f"{terms} = {b[i]:.3f}")

    # Backward substitution
    x = np.zeros(n)
    print("\n=== Backward Substitution ===")
    for i in range(n - 1, -1, -1):
        if A[i, i] == 0:
            raise ValueError("Zero on diagonal - no unique solution")
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        print(f"x[{i}] = {x[i]:.4f}")

    print("\nFinal Solution:")
    print(x)
    return x
