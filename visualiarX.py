import numpy as np

X = np.load("data/X.npy")

for i in range(5):
    print(f"\n=== Amostra {i} ===")

    matriz = X[i].squeeze()

    for linha in matriz:
        print("".join("█" if v == -1 else " " for v in linha))