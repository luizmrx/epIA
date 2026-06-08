import numpy as np

from mlp import MLP

mlp = MLP(
    input_size=2,
    hidden_size=3,
    output_size=2,
    learning_rate=0.5
)

mlp.W1 = np.array([
    [ 0.1,  0.1, -0.1],
    [-0.1,  0.1, -0.1]
])

mlp.W2 = np.array([
    [ 0.1, -0.1],
    [ 0.0,  0.1],
    [ 0.1, -0.1]
])

mlp.b1 = np.array([
    [-0.1, -0.1, 0.1]
])

mlp.b2 = np.array([
    [-0.1, 0.1]
])

X = np.array([[1,1]])

Y = np.array([[1,0]])

output = mlp.forward(X)

print("Hidden:")
print(mlp.a_hidden)

print("\nOutput:")
print(output)

mlp.backward(X,Y)

print("\nNovos W1:")
print(mlp.W1)

print("\nNovos W2:")
print(mlp.W2)