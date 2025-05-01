import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util

imagen_ruido = io.imread('Ruido/imagenes/nutria.jpg')
imagen_original = io.imread('Ruido/imagenes/nutria_grises.jpg')

imagen_ruido = imagen_ruido / 255.0
imagen_original = imagen_original / 255.0

imagen_ruido = color.rgb2gray(imagen_ruido)
imagen_original = color.rgb2gray(imagen_original)


def grad_mse(x):
    return 2 * (x - imagen_original)

def caso_real(x):
    return np.sum((x - imagen_original) ** 2)

def energia_J(x, ruido, lam):
    fidelity = 0.5 * np.sum((x - ruido) ** 2)
    
    grad_x = x[1:, :] - x[:-1, :]
    grad_y = x[:, 1:] - x[:, :-1]
    smoothness = 0.5 * lam * (np.sum(grad_x ** 2) + np.sum(grad_y ** 2))
    
    return fidelity + smoothness

def gradiente_J(x, ruido, lam):
    grad = x - ruido

    grad[:-1, :] += lam * (x[:-1, :] - x[1:, :])
    grad[1:, :]  += lam * (x[1:, :] - x[:-1, :])
    
    grad[:, :-1] += lam * (x[:, :-1] - x[:, 1:])
    grad[:, 1:]  += lam * (x[:, 1:] - x[:, :-1])
    
    return grad



def descenso_grad(grad_f, x0,ruido,lam, lr, max_ite, eps):
    for i in range(max_ite):
        gradiente = grad_f(x0,ruido, lam)
        norma = np.linalg.norm(gradiente)
        print(f"Iteración {i+1}: norma del gradiente = {norma}")
        if norma < eps:
            break
        x0 = x0 - lr * gradiente
    return x0

max_iter = 1000
eps = 1e-8
lr = 0.03
lam = 0.1

imagen_pred = np.random.rand(*imagen_ruido.shape)

imagen_pred = descenso_grad(gradiente_J, imagen_pred, imagen_ruido,lam, lr, max_iter, eps)

plt.imshow(imagen_pred, cmap='gray')
plt.title('Imagen Denoised')
plt.axis('off')
plt.show()
