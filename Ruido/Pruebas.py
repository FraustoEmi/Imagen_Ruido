#Aqui se haran pruebas de las funciones
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from scipy.ndimage import gaussian_filter

img = io.imread('Ruido/imagenes/nutria.jpg')
plt.imshow(img, cmap='gray')
plt.show()
