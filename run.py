import numpy as np

# ==========================================
# ETAPA 1: FUNÇÕES MATEMÁTICAS BASE
# ==========================================

def sigmoid(z):
    """Aplica a função Sigmoide para espremer os valores entre 0 e 1."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """
    Calcula a derivada da Sigmoide utilizando a própria saída 'a' do neurônio.
    Nota: O parâmetro 'a' já deve ser o valor após passar pela sigmoide (a = sigmoid(z)).
    """
    return a * (1 - a)

# ==========================================
# ETAPA 2 e 3: CONFIGURAÇÃO DA ARQUITETURA
# ==========================================

# Dados de treino do XOR (Entradas X e Gabarito Y)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]) # Dimensão: 4x2 (4 exemplos de treino, 2 entradas cada)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
]) # Dimensão: 4x1 (Gabarito correto para cada um dos 4 exemplos)

# Configuração da semente aleatória para que os resultados sejam reproduzíveis
np.random.seed(42)

# Hiperparâmetros
learning_rate = 0.5  # Taxa de aprendizado (tamanho do passo no Gradiente Descendente)
epochs = 20000       # Quantas vezes a rede vai olhar para os dados e tentar aprender

# Inicialização dos Pesos e Bias com valores aleatórios pequenos
# Camada Escondida (2 entradas -> 2 neurônios escondidos)
W_hidden = np.random.uniform(size=(2, 2))  # Matriz 2x2
B_hidden = np.zeros((1, 2))                # Vetor 1x2 (inicializado com 0)

# Camada de Saída (2 neurônios escondidos -> 1 neurônio de saída)
W_output = np.random.uniform(size=(2, 1))  # Matriz 2x1
B_output = np.zeros((1, 1))                # Vetor 1x1 (inicializado com 0)

print("Treinando a rede neural...")

# ==========================================
# ETAPA 4: LOOP DE TREINO (FEEDFORWARD + BACKPROPAGATION)
# ==========================================

for epoch in range(epochs):
    
    # --------------------------------------
    # 4.1 FEEDFORWARD (Propagação para Frente)
    # --------------------------------------
    
    # Cálculo da Camada Escondida
    Z_hidden = np.dot(X, W_hidden) + B_hidden  # Soma ponderada bruta (Etapa 1.1)
    A_hidden = sigmoid(Z_hidden)               # Ativação via Sigmoide (Etapa 1.2)
    
    # Cálculo da Camada de Saída
    Z_output = np.dot(A_hidden, W_output) + B_output  # Nova soma ponderada com os dados da oculta
    Y_hat = sigmoid(Z_output)                         # Previsão final da rede (ŷ)
    
    # Cálculo do Erro Quadrático (Opcional: apenas para monitorar o aprendizado)
    if epoch % 2000 == 0:
        error = np.mean(0.5 * (Y - Y_hat) ** 2)  # Fórmula do Erro da Rede (Etapa 1.4)
        print(f"Época {epoch:5d} | Erro Médio: {error:.6f}")
        
    # --------------------------------------
    # 4.2 BACKPROPAGATION (Cálculo da "Culpa" do Erro)
    # --------------------------------------
    
    # Passo 1: Descobrir o erro na camada de saída (delta_out)
    # Diferença (Previsão - Real) multiplicada pela inclinação (derivada) da curva na saída
    error_output_layer = Y_hat - Y
    delta_output = error_output_layer * sigmoid_derivative(Y_hat)
    
    # Passo 2: Retropropagar o erro para a camada escondida (delta_hidden)
    # O erro escondido depende do erro de saída multiplicado pelos pesos que conectavam as duas
    error_hidden_layer = np.dot(delta_output, W_output.T)
    delta_hidden = error_hidden_layer * sigmoid_derivative(A_hidden)
    
    # --------------------------------------
    # 4.3 GRADIENTE DESCENDENTE (Atualização dos Pesos)
    # --------------------------------------
    
    # peso_novo = peso_antigo - (learning_rate * gradiente)
    # Usamos o produto escalar com a transposta (.T) para alinhar as dimensões das matrizes
    W_output -= learning_rate * np.dot(A_hidden.T, delta_output)
    B_output -= learning_rate * np.sum(delta_output, axis=0, keepdims=True)
    
    W_hidden -= learning_rate * np.dot(X.T, delta_hidden)
    B_hidden -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

print("\nTreinamento concluído!")
print("-" * 40)

# ==========================================
# TESTANDO A REDE APÓS O TREINO
# ==========================================
print("Resultados do Teste Final:")

# Executa um último Feedforward para ver o que ela aprendeu
Z_hidden_final = np.dot(X, W_hidden) + B_hidden
A_hidden_final = sigmoid(Z_hidden_final)
Z_output_final = np.dot(A_hidden_final, W_output) + B_output
Y_final = sigmoid(Z_output_final)

for i in range(len(X)):
    print(f"Entrada: {X[i]} -> Previsão da IA: {Y_final[i][0]:.4f} (Gabarito: {Y[i][0]})")