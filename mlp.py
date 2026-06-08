"""
Luiz Souza Neto - 15473503

André Tanikawa de Oliveira - 15048766

Pedro Teixeira Guimarães - 12821669

Gustavo Jun Arakaki - 14028786

Renan Biruel Uema - 15486661

Renan Mochizuki - 15579831
"""

"""
Implementação de uma Rede Neural Perceptron Multicamadas (MLP).

Características:
- Uma camada oculta
- Função de ativação Sigmoide
- Treinamento por Backpropagation
- Gradiente Descendente
- Inicialização Xavier

A rede pode ser utilizada tanto para problemas de classificação binária quanto multiclasse.
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

        # Inicialização Xavier.
        #
        # Os pesos são inicializados com valores aleatórios
        # dentro de um intervalo calculado a partir do número
        # de neurônios das camadas conectadas.
        #
        # Essa estratégia ajuda a evitar saturação precoce
        # dos neurônios e melhora a estabilidade do treinamento.

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

        # Vetores de bias.
        #
        # Permitem deslocar a função de ativação,
        # aumentando a capacidade de representação da rede.

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
        Executa a propagação para frente (Forward Propagation).

        Fluxo:

        Entrada
        ↓
        Camada Oculta
        ↓
        Sigmoide
        ↓
        Camada de Saída
        ↓
        Sigmoide

        Retorna as saídas produzidas pela rede.
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
        """
        Executa o algoritmo de Backpropagation.

        Calcula os gradientes dos pesos e bias a partir
        do erro produzido pela rede e realiza a atualização
        dos parâmetros utilizando Gradiente Descendente.
        """

        n_samples = X.shape[0]

        # --------------------------
        # Camada de saída
        # --------------------------

        # Diferença entre a saída desejada e a saída produzida.
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
        # Atualização baseada no gradiente médio do lote.

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
        Calcula o Erro Quadrático Médio (MSE).

        Utilizado para medir a diferença entre
        a saída esperada e a saída produzida pela rede.
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
        X_val=None,
        Y_val=None,
        epochs=1000,
        patience=50,
        target_error=0.001,
        verbose=True
    ):

        history = []

        best_val_error = float("inf")
        best_weights = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):

            # Forward + treino
            self.forward(X)
            error = self.compute_error(Y)
            self.backward(X, Y)
            history.append(error)

            # --------------------------
            # Validação
            # --------------------------
            if X_val is not None:

                val_output = self.forward(X_val)

                val_error = np.mean(
                    (Y_val - val_output) ** 2
                )

                # salvar melhor modelo
                if val_error < best_val_error:

                    best_val_error = val_error
                    best_weights = (
                        self.W1.copy(),
                        self.W2.copy(),
                        self.b1.copy(),
                        self.b2.copy()
                    )

                    best_epoch = epoch
                    patience_counter = 0

                else:
                    patience_counter += 1

                # log
                if verbose and epoch % 100 == 0:
                    print(
                        f"Epoch {epoch:5d} "
                        f"| Train={error:.6f} "
                        f"| Val={val_error:.6f}"
                    )

                # --------------------------
                # EARLY STOPPING (patience)
                # Interrompe o treinamento caso o erro de validação
                # deixe de melhorar por várias épocas consecutivas.
                # --------------------------
                if patience_counter >= patience:
                    print("\nEarly stopping acionado!")
                    print(f"Melhor época: {best_epoch}")
                    print(f"Val Error: {best_val_error:.6f}")
                    break

            # --------------------------
            # Parada por erro mínimo (opcional)
            # --------------------------
            if error <= target_error:
                print("\nErro mínimo atingido!")
                break

        # restaurar melhores pesos
        if best_weights is not None:
            self.W1, self.W2, self.b1, self.b2 = best_weights

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
        """
        Exibe um resumo da arquitetura e dos
        principais hiperparâmetros da rede.
        """

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