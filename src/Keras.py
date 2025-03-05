import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

def train_mnist_model():
    # Cargar datos de entrenamiento y prueba desde MNIST
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Visualizar un ejemplo de imagen de entrenamiento
    plt.imshow(train_data_x[1], cmap="gray")
    plt.title("Ejemplo de imagen de entrenamiento")
    plt.show()

    # Normalización de datos
    x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
    y_train = to_categorical(train_labels_y)
    x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255
    y_test = to_categorical(test_labels_y)

    # Definir arquitectura de la red neuronal
    model = Sequential([
        Input(shape=(28*28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(x_train, y_train, epochs=8, batch_size=128)

    # Evaluar el modelo
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

if __name__ == "__main__":
    train_mnist_model()
