from mlp import MLP
import numpy as np

def test_mesa():

    print("\n" + "=" * 50)
    print("TESTE DE MESA")
    print("=" * 50)

    mlp = MLP(
        input_size=2,
        hidden_size=3,
        output_size=2,
        learning_rate=0.5
    )

    # pesos da camada escondida

    mlp.W1 = np.array([
        [ 0.1,  0.1, -0.1],
        [-0.1,  0.1, -0.1]
    ])

    mlp.b1 = np.array([
        [-0.1, -0.1, 0.1]
    ])

    # pesos da saída

    mlp.W2 = np.array([
        [ 0.1, -0.1],
        [ 0.0,  0.1],
        [ 0.1, -0.1]
    ])

    mlp.b2 = np.array([
        [-0.1, 0.1]
    ])

    X = np.array([[1,1]])
    Y = np.array([[1,0]])

    output = mlp.forward(X)

    print("\nSaída camada escondida:")
    print(mlp.a_hidden)

    print("\nSaída final:")
    print(output)

    mlp.backward(X, Y)

    print("\nPesos atualizados W1:")
    print(mlp.W1)

    print("\nPesos atualizados W2:")
    print(mlp.W2)