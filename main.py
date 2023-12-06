import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Base de Dados
celsius = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit = np.array([-40,  14, 32, 46.4, 59, 71.6, 100], dtype=float)

# Gerando 1000 valores aleatórios para Celsius e Fahrenheit
np.random.seed(42)  # Define a semente para reprodução dos valores aleatórios
num_samples = 1000

celsius_random = np.random.uniform(low=-100.0, high=100.0, size=num_samples)
fahrenheit_random = celsius_random * 1.8 + 32

# Adicionando os novos valores aleatórios aos conjuntos existentes
celsius = np.concatenate((celsius, celsius_random))
fahrenheit = np.concatenate((fahrenheit, fahrenheit_random))

# Definindo e treinando o modelo
l1 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l1])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

historico = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Treino Finalizado!")

# Visualização do treinamento
losses = historico.history['loss']
epochs = range(1, len(losses) + 1)

plt.plot(epochs, losses, 'b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Fazendo previsão e mostrando os pesos da camada
input_celsius = float(input("Digite a temperatura em Celsius para a previsão: "))
previsao = model.predict([input_celsius])
print("A previsão em Fahrenheit é:", previsao)
print("Estes são os pesos da camada: {}".format(l1.get_weights()))
