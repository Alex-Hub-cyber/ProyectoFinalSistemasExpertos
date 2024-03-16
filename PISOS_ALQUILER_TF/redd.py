import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

"""
ciudad (0), m^2, nº hab, nº planta, ascensor (0-1), exterior (0-1), 
estado (0 no rehabilitado, 1 rehab, 2 nuevo), céntrico (0, 1)
"""

features = [(0, 54, 2, 4, 0, 1, 0, 0),
            (0, 152, 2, 4, 1, 1, 3, 1),
            (0, 64, 3, 4, 0, 1, 0, 0),
            (0, 154, 5, 4, 1, 1, 1, 1),
            (0, 100, 1, 5, 1, 1, 1, 0),
            (0, 140, 5, 2, 1, 1, 2, 0),
            (0, 120, 3, 2, 1, 1, 1, 1),
            (0, 70, 2, 3, 1, 1, 1, 0),
            (0, 60, 2, 2, 0, 1, 1, 1),
            (0, 129, 3, 18, 1, 1, 2, 1),
            (0, 93, 1, 3, 1, 1, 2, 0),
            (0, 52, 2, 2, 0, 1, 1, 1),
            (0, 110, 3, 5, 1, 1, 1, 1),
            (0, 63, 3, 2, 1, 1, 1, 0),
            (0, 160, 1, 4, 1, 1, 2, 0)
            ]
targets = [750, 2000, 650, 1500, 900, 1000, 1300, 750, 900, 1800, 975, 880, 1400, 750, 1050]

# Normalizar los datos de entrada y salida
#features, targets = tf.keras.utils.normalize(features, targets)
features = np.array(features)
targets = np.array(targets)

# Crear un modelo secuencial
model = tf.keras.Sequential()
 
# Añadir una capa densa con 16 neuronas y función de activación ReLU
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)))

# Añadir una capa de dropout con una tasa del 20%
model.add(tf.keras.layers.Dropout(0.2))

# Añadir una capa densa con 8 neuronas y función de activación ReLU
model.add(tf.keras.layers.Dense(8, activation='relu'))

# Añadir una capa de dropout con una tasa del 20%
model.add(tf.keras.layers.Dropout(0.2))

# Añadir una capa densa con 1 neurona y función de activación lineal
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compilar el modelo con optimizador Adam, función de pérdida error cuadrático medio y métrica error absoluto medio
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error',
    metrics = ['mean_absolute_error']
)

# Crear un callback de parada temprana que detenga el entrenamiento cuando la pérdida de validación no mejore en 10 épocas
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Crear un callback de punto de control que guarde el mejor modelo según la pérdida de validación
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('pisos_alquiler.keras', monitor='val_loss', save_best_only=True)

print('Inicio de entrenamiento...')
# Entrenar el modelo con 1000 épocas, un 20% de datos de validación y los callbacks creados
historial = model.fit(features, targets, epochs=1000, validation_split=0.2, callbacks=[early_stopping, model_checkpoint], verbose=False)
print('Modelo entrenado!')

# Mostrar la gráfica de la pérdida por época
plt.xlabel('#Época')
plt.ylabel('Error cuadrático medio')
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.legend()
plt.show()

model.save('pisos_alquiler.keras')
