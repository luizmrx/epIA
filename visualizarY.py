import numpy as np

y = np.load("data/Y_classe.npy", allow_pickle=True)
for i in range(10):
    print(y[i])