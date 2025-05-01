#Aqui se haran pruebas de las funciones
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from scipy.ndimage import gaussian_filter

imagen = io.imread('Ruido/imagenes/nutria.jpg')
plt.imshow(imagen, cmap='gray')

#Esta imagen tiene 3 canales (RGB), por lo que es conveniente convertirla a escala de grises
#para evitar problemas de compatibilidad con las funciones de ruido

imagen_grises = color.rgb2gray(imagen)
plt.imshow(imagen_grises, cmap='gray')

#plt.imsave('Ruido/imagenes/nutria_grises.jpg', imagen_grises, cmap='gray')

#Agregamos ruido gaussiano a la imagen
imagen_ruido = util.random_noise(imagen_grises, mode='gaussian', var=0.9)
plt.imshow(imagen_ruido, cmap='gray')
plt.show()

plt.imsave('Ruido/imagenes/nutria_ruido.jpg', imagen_ruido, cmap='gray')

MSE = np.mean((imagen_grises - imagen_ruido) ** 2)
print("MSE:", MSE)

# SE TIENE QUE REDUCIR EL RUIDO DE LA IMAGEN (MSE)