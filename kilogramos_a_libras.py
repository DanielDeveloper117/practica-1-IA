import tensorflow as tf
import numpy as np

kilogramos = np.array([0, 1, 2, 3, 4, 5], dtype=float)
libras = np.array([0, 2.20462, 4.40924, 6.61386, 8.81848, 11.0231], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(kilogramos, libras, epochs=1000, verbose=False)
print("Modelo entrenado con éxito!")

import matplotlib.pyplot as plt

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Realizar una predicción")
resultado = modelo.predict([10])
print("El resultado es: " + str(resultado) + " libras")

modelo.save('kilogramos_a_libras.h5')