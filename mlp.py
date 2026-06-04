"""
mlp.py

Implementação de uma Rede Neural Artificial
Multilayer Perceptron (MLP)

Características:

- Uma camada escondida
- Função de ativação Sigmoid
- Backpropagation
- Gradiente Descendente
- Xavier Initialization
"""

import numpy as np

from activation import sigmoid
from activation import sigmoid_derivative


class MLP:

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.1,
        seed=42
    ):
        """
        Inicializa a arquitetura da rede.
        """

        np.random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate

        # ==================================================
        # XAVIER INITIALIZATION
        # ==================================================

        limit_hidden = np.sqrt(
            6 / (input_size + hidden_size)
        )

        limit_output = np.sqrt(
            6 / (hidden_size + output_size)
        )

        self.W1 = np.random.uniform(
            -limit_hidden,
            limit_hidden,
            (input_size, hidden_size)
        )

        self.W2 = np.random.uniform(
            -limit_output,
            limit_output,
            (hidden_size, output_size)
        )

        # ==================================================
        # BIAS
        # ==================================================

        self.b1 = np.zeros(
            (1, hidden_size)
        )

        self.b2 = np.zeros(
            (1, output_size)
        )

        # ==================================================
        # SALVA PESOS INICIAIS
        # ==================================================

        self.initial_W1 = self.W1.copy()
        self.initial_W2 = self.W2.copy()

        self.initial_b1 = self.b1.copy()
        self.initial_b2 = self.b2.copy()

        # ==================================================
        # HISTÓRICO
        # ==================================================

        self.error_history = []

    # ==================================================
    # FORWARD
    # ==================================================

    def forward(self, X):
        """
        Propagação para frente.
        """

        self.z_hidden = np.dot(
            X,
            self.W1
        ) + self.b1

        self.a_hidden = sigmoid(
            self.z_hidden
        )

        self.z_output = np.dot(
            self.a_hidden,
            self.W2
        ) + self.b2

        self.a_output = sigmoid(
            self.z_output
        )

        return self.a_output

    # ==================================================
    # BACKPROPAGATION
    # ==================================================

    def backward(self, X, Y):

        n_samples = X.shape[0]

        # --------------------------
        # Camada de saída
        # --------------------------

        output_error = (
            Y - self.a_output
        )

        output_delta = (
            output_error *
            sigmoid_derivative(
                self.a_output
            )
        )

        # --------------------------
        # Camada escondida
        # --------------------------

        hidden_error = np.dot(
            output_delta,
            self.W2.T
        )

        hidden_delta = (
            hidden_error *
            sigmoid_derivative(
                self.a_hidden
            )
        )

        # --------------------------
        # Atualização W2
        # --------------------------

        self.W2 += (
            self.learning_rate *
            np.dot(
                self.a_hidden.T,
                output_delta
            ) / n_samples
        )

        self.b2 += (
            self.learning_rate *
            np.sum(
                output_delta,
                axis=0,
                keepdims=True
            ) / n_samples
        )

        # --------------------------
        # Atualização W1
        # --------------------------

        self.W1 += (
            self.learning_rate *
            np.dot(
                X.T,
                hidden_delta
            ) / n_samples
        )

        self.b1 += (
            self.learning_rate *
            np.sum(
                hidden_delta,
                axis=0,
                keepdims=True
            ) / n_samples
        )

    # ==================================================
    # ERRO
    # ==================================================

    def compute_error(self, Y):
        """
        Mean Squared Error.
        """

        return np.mean(
            (Y - self.a_output) ** 2
        )

    # ==================================================
    # TREINAMENTO
    # ==================================================

    def train(
        self,
        X,
        Y,
        epochs=1000,
        target_error=0.001,
        verbose=True
    ):
        """
        Treina a rede usando
        Backpropagation.
        """

        history = []

        for epoch in range(epochs):

            self.forward(X)

            error = self.compute_error(Y)

            self.backward(X, Y)

            history.append(error)

            if (
                verbose
                and epoch % 100 == 0
            ):
                print(
                    f"Epoch {epoch:5d} "
                    f"| Erro = {error:.6f}"
                )

            # Early Stopping

            if error <= target_error:

                print(
                    f"\nTreinamento encerrado "
                    f"na época {epoch}"
                )

                print(
                    f"Erro final = {error:.6f}"
                )

                break

        self.error_history = history

        return history

    # ==================================================
    # PREDIÇÃO
    # ==================================================

    def predict(self, X):

        return self.forward(X)

    # ==================================================
    # CLASSIFICAÇÃO
    # ==================================================

    def predict_class(self, X):
        """
        Retorna a classe prevista.
        """

        output = self.forward(X)

        # Multi-classe

        if self.output_size > 1:

            return np.argmax(
                output,
                axis=1
            )

        # Binário

        return np.where(
            output >= 0.5,
            1,
            0
        )

    # ==================================================
    # LETRAS
    # ==================================================

    def predict_letters(
        self,
        X,
        index_to_label
    ):
        """
        Converte índices em letras.
        """

        prediction = self.predict_class(X)

        return [
            index_to_label[idx]
            for idx in prediction
        ]

    # ==================================================
    # SALVAR PESOS
    # ==================================================

    def save_weights(
        self,
        filename_prefix="weights"
    ):
        """
        Salva pesos e bias.
        """

        np.savetxt(
            f"{filename_prefix}_W1.csv",
            self.W1,
            delimiter=","
        )

        np.savetxt(
            f"{filename_prefix}_W2.csv",
            self.W2,
            delimiter=","
        )

        np.savetxt(
            f"{filename_prefix}_b1.csv",
            self.b1,
            delimiter=","
        )

        np.savetxt(
            f"{filename_prefix}_b2.csv",
            self.b2,
            delimiter=","
        )

    # ==================================================
    # RESUMO
    # ==================================================

    def summary(self):

        print("\n===== MLP =====")

        print(
            f"Entradas: {self.input_size}"
        )

        print(
            f"Oculta: {self.hidden_size}"
        )

        print(
            f"Saídas: {self.output_size}"
        )

        print(
            f"Learning Rate: {self.learning_rate}"
        )

        print("================\n")