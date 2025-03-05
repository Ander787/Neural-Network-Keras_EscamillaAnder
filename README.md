# README - Red Neuronal Keras

## 🤓 Información del Proyecto
**Materia:** Redes Neuronales  
**Tarea:** Clasificación de dígitos utilizando el conjunto de datos MNIST  
**Estudiante:** Ander Heinrich Escamilla Wong  

**Fecha:** 03/03/2025  

---

## ✏️ Descripción General
Este repositorio contiene un script en Python que entrena una red neuronal para clasificar dígitos escritos a mano utilizando el conjunto de datos MNIST. La red neuronal se implementa con Keras y consta de dos capas densas. El modelo se entrena durante 8 épocas y se evalúa en un conjunto de prueba.

---

## 🧠 Arquitectura de la Red Neuronal
El modelo tiene la siguiente estructura:
- Capa de entrada con 28x28 píxeles (dimensión de las imágenes MNIST).
- Una capa oculta densa con 512 neuronas y activación ReLU.
- Una capa de salida con 10 neuronas (una para cada dígito del 0 al 9) y activación softmax.

---

## ☝️ Requisitos Previos
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas en tu entorno de Python:

```sh
pip install numpy keras matplotlib
```
## 📋 Estructura del Repositorio
```
mnist_clasificacion/
│── train_mnist_model.py     # Código principal para entrenar y evaluar el modelo
│── README.md                # Este documento
```
### 1. train_mnist_model.py
Este archivo contiene el código principal para cargar los datos, entrenar el modelo y evaluar su rendimiento en el conjunto de prueba.

### 2. Cargar y Preprocesar Datos
```python
(train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
```

### 3. Visualizar una imagen
Se muestra una imagen de ejemplo del conjunto de entrenamiento para tener una idea del tipo de datos con los que se trabaja.
```python
plt.imshow(train_data_x[1], cmap="gray")
plt.title("Ejemplo de imagen de entrenamiento")
plt.show()
```

### 4. Normalización de Datos
Las imágenes se redimensionan de 28x28 píxeles a un vector de 784 dimensiones (28*28). Además, se normalizan dividiendo entre 255 para que los valores de píxeles estén entre 0 y 1.
```python
x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
y_train = to_categorical(train_labels_y)
x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255
y_test = to_categorical(test_labels_y)
```

### 5. Definir la Red Neuronal
Se define un modelo secuencial que consiste en:
-Capa de entrada: 28x28 píxeles de entrada.
-Capa oculta: 512 neuronas con la función de activación ReLU.
-Capa de salida: 10 neuronas con activación softmax para clasificación de 10 clases (dígitos del 0 al 9).
```python
model = Sequential([
    Input(shape=(28*28,)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])
```

### 6. Compilación y Entrenamiento del Modelo
El modelo se compila utilizando el optimizador RMSProp y la función de pérdida categorical_crossentropy (ya que es una clasificación multiclase).
Entrenamiento: El modelo se entrena durante 8 épocas utilizando un tamaño de lote de 128 imágenes.
```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=8, batch_size=128)
```

### 7. Evaluación del modelo
Se evalúa el modelo en el conjunto de prueba y se imprime la precisión alcanzada en los datos de prueba.
```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=8, batch_size=128)
```

## 📖 Uso
Para ejecutar la red neuronal, usa el siguiente comando:

```sh
python main.py
```

Esto entrenará el modelo sobre el conjunto de datos MNIST y mostrará la precisión en el conjunto de prueba.

## 📊 Resultados Esperados

El programa generará:
-Visualización de una imagen de entrenamiento del conjunto MNIST.
-Precisión del modelo en el conjunto de prueba, que se imprimirá en consola después de entrenar.

## 🔍 Conclusión

Este proyecto entrena y evalúa una red neuronal simple para la clasificación de dígitos en el conjunto de datos MNIST utilizando Keras, proporcionando una forma sencilla de aplicar redes neuronales en tareas de clasificación.

## ⭐ ¡Dale una estrella al repo si te fue útil!
