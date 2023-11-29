import tensorflow as tf
import numpy as np


celcius=np.array([-15, -5, 0, 5, 15], dtype=float)
farenheit=np.array([5,23,32,41,59], dtype=float)

#capa=tf.keras.layers.Dense(units=1,input_shape=[1])
#modelo=tf.keras.Sequential([capa])
oculta1=tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2=tf.keras.layers.Dense(units=3)
salida=tf.keras.layers.Dense(units=1)
modelo=tf.keras.Sequential([oculta1,oculta2,salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
print("Comenzando entrenamiento...")

historial=modelo.fit(celcius,farenheit,epochs=1000,verbose=False)
print("Modelo entrenado.")


import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print("Realizar una prediccion")
resultado=modelo.predict([100,0])
print("El resultado es: "+str(resultado)+"F")

modelo.save('celcius_a_farenheigt.h5')
