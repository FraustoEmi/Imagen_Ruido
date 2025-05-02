import time
import funciones as fn
from skimage import io, color, util
import numpy as np
import matplotlib.pyplot as plt

#****************************cargar imagen y añadir ruido****************************
ruido = 0.9
ruido2 = 0.45
ruido3 = 0.1
imagen_grises, imagen_ruido = fn.añadir_ruido('Ruido/imagenes/nutria.jpg', ruido)
imagen_grises2, imagen_ruido2 = fn.añadir_ruido('Ruido/imagenes/nutria.jpg', ruido2)
imagen_grises3, imagen_ruido3 = fn.añadir_ruido('Ruido/imagenes/nutria.jpg', ruido3)

#********************Verificar que esten a escala de grises y normalizar*************
#imagen_grises = color.rgb2gray(imagen_grises)
#imagen_ruido = color.rgb2gray(imagen_ruido)

imagen_grises = imagen_grises / 255.0
imagen_ruido = imagen_ruido / 255.0

#imagen_grises2 = color.rgb2gray(imagen_grises2)
#imagen_ruido2 = color.rgb2gray(imagen_ruido2)

imagen_grises2 = imagen_grises2 / 255.0
imagen_ruido2 = imagen_ruido2 / 255.0

#imagen_grises3 = color.rgb2gray(imagen_grises3)
#imagen_ruido3 = color.rgb2gray(imagen_ruido3)

imagen_grises3 = imagen_grises3 / 255.0
imagen_ruido3 = imagen_ruido3 / 255.0

#***********************Uso de las funciones, tomando el tiempos*********************
#Parametros de la tabla
max_iter = 1000
eps = 1e-6
lr = 0.001
lam = 0.1
eta=0.9

#Imagen inicial aleatoria
imagen_pred = np.random.rand(*imagen_ruido.shape)
imagen_pred2 = np.random.rand(*imagen_ruido2.shape)
imagen_pred3 = np.random.rand(*imagen_ruido3.shape)

#*************************Descenso de gradiente, momentum y nesterov****************
inicio_1 = time.time()
imagen_pred, error_simple, iteracion_simple = fn.descenso_grad(fn.gradiente_J, imagen_pred, imagen_ruido,lam, lr, max_iter, eps)
fin_1 = time.time()
inicio_2 = time.time()
imagen_pred2, error_momentum, iteracion_momentum = fn.momentum(fn.gradiente_J, imagen_pred, imagen_ruido,lam, lr, max_iter, eps,eta)
fin_2 = time.time()
inicio_3 = time.time()
imagen_pred3, error_nesterov, iteracion_nesterov = fn.nesterov(fn.gradiente_J, imagen_pred, imagen_ruido,lam, lr, max_iter, eps,eta)
fin_3 = time.time()

inicio_4 = time.time()
imagen_pred4, error_simple2, iteracion_simple2 = fn.descenso_grad(fn.gradiente_J, imagen_pred2, imagen_ruido2,lam, lr, max_iter, eps)
fin_4 = time.time()
inicio_5 = time.time()
imagen_pred5, error_momentum2, iteracion_momentum2 = fn.momentum(fn.gradiente_J, imagen_pred2, imagen_ruido2,lam, lr, max_iter, eps,eta)
fin_5 = time.time()
inicio_6 = time.time()
imagen_pred6, error_nesterov2, iteracion_nesterov2 = fn.nesterov(fn.gradiente_J, imagen_pred2, imagen_ruido2,lam, lr, max_iter, eps,eta)
fin_6 = time.time()

inicio_7 = time.time()
imagen_pred7, error_simple3, iteracion_simple3 = fn.descenso_grad(fn.gradiente_J, imagen_pred3, imagen_ruido3,lam, lr, max_iter, eps)
fin_7 = time.time()
inicio_8 = time.time()
imagen_pred8, error_momentum3, iteracion_momentum3 = fn.momentum(fn.gradiente_J, imagen_pred3, imagen_ruido3,lam, lr, max_iter, eps,eta)
fin_8 = time.time()
inicio_9 = time.time()
imagen_pred9, error_nesterov3, iteracion_nesterov3 = fn.nesterov(fn.gradiente_J, imagen_pred3, imagen_ruido3,lam, lr, max_iter, eps,eta)
fin_9 = time.time()

#********************************Imprimir resultados*****************************
print('\nGradiente \t\tRuido \titeraciones \tTiempo  \terror')
print('Descenso de Gradiente \t', ruido, '\t', iteracion_simple, '\t', fin_1 - inicio_1, '\t', error_simple[-1])
print('Momentum \t\t', ruido, '\t', iteracion_momentum, '\t', fin_2 - inicio_2, '\t', error_momentum[-1])
print('Nesterov \t\t', ruido, '\t', iteracion_nesterov, '\t', fin_3 - inicio_3, '\t', error_nesterov[-1])

print('\nGradiente \t\tRuido \titeraciones \tTiempo  \terror')
print('Descenso de Gradiente \t', ruido2, '\t', iteracion_simple2, '\t', fin_4 - inicio_4, '\t', error_simple2[-1])
print('Momentum \t\t', ruido2, '\t', iteracion_momentum2, '\t', fin_5 - inicio_5, '\t', error_momentum2[-1])
print('Nesterov \t\t', ruido2, '\t', iteracion_nesterov2, '\t', fin_6 - inicio_6, '\t', error_nesterov2[-1])

print('\nGradiente \t\tRuido \titeraciones \tTiempo  \terror')
print('Descenso de Gradiente \t', ruido3, '\t', iteracion_simple3, '\t', fin_7 - inicio_7, '\t', error_simple3[-1])
print('Momentum \t\t', ruido3, '\t', iteracion_momentum3, '\t', fin_5 - inicio_5, '\t', error_momentum3[-1])
print('Nesterov \t\t', ruido3, '\t', iteracion_nesterov3, '\t', fin_6 - inicio_6, '\t', error_nesterov3[-1])

#*********************************Mostrar la tabla de imagenes*********************************
import matplotlib.pyplot as plt

# Supongamos que ya tienes estas listas definidas:

titulos_columnas = ['Original', 'Gradiente', 'Momentum', 'Nesterov']

imagenes_ruido = [
    imagen_ruido,imagen_ruido2, imagen_ruido3
]

predicciones = [
    [imagen_pred, imagen_pred2, imagen_pred3],
    [imagen_pred4, imagen_pred5, imagen_pred6],
    [imagen_pred7, imagen_pred8, imagen_pred9]
]

fig, ax = plt.subplots(3, 4, figsize=(16, 12))

for i in range(3):  # Filas (tipos de ruido)
    for j in range(4):  # Columnas (métodos)
        if j == 0:
            ax[i, j].imshow(imagenes_ruido[i], cmap='gray')
            ax[i, j].set_title('Ruido')
        else:
            ax[i, j].imshow(predicciones[i][j-1], cmap='gray')
            ax[i, j].set_title(titulos_columnas[j])
        ax[i, j].axis('off')

plt.tight_layout()
plt.show()

#*********************************Mostrar el error*********************************
fig, ax = plt.subplots(3, 3, figsize=(10, 15))
ax[0,0].plot(error_simple, label='Descenso de Gradiente')
ax[0,0].legend()
ax[0,1].plot(error_momentum, label='Momentum')
ax[0,1].legend()
ax[0,2].plot(error_nesterov, label='Nesterov')
ax[0,2].legend()
ax[1,0].plot(error_simple2, label='Descenso de Gradiente')
ax[1,0].legend()
ax[1,1].plot(error_momentum2, label='Momentum')
ax[1,1].legend()
ax[1,2].plot(error_nesterov2, label='Nesterov')
ax[1,2].legend()
ax[2,0].plot(error_simple3, label='Descenso de Gradiente')
ax[2,0].legend()
ax[2,1].plot(error_momentum3, label='Momentum')
ax[2,1].legend()
ax[2,2].plot(error_nesterov3, label='Nesterov')
ax[2,2].legend()
plt.show()