import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util

def añadir_ruido(url, cnt_ruido):
    img = io.imread(url)
    imagen_grises = color.rgb2gray(img)
    imagen_ruido = util.random_noise(imagen_grises, mode='gaussian', var=cnt_ruido)
    
    plt.imsave('Ruido/imagenes/nutria_grises.jpg', imagen_grises, cmap='gray')
    plt.imsave('Ruido/imagenes/nutria_ruido.jpg', imagen_ruido, cmap='gray')
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Imagen Original')
    ax[0].axis('off')
    ax[1].imshow(imagen_grises, cmap='gray')
    ax[1].set_title('Imagen a Grises')
    ax[1].axis('off')
    ax[2].imshow(imagen_ruido, cmap='gray')
    ax[2].set_title('Imagen con Ruido')
    ax[2].axis('off')
    plt.show()
    
    return imagen_grises, imagen_ruido

def descenso_grad(grad_f, x0,ruido,lam, lr, max_ite, eps):
    lista_normas = []
    iteracion = 0
    for i in range(max_ite):
        gradiente = grad_f(x0,ruido, lam)
        norma = np.linalg.norm(gradiente)
        lista_normas.append(norma)
        if norma < eps:
            break
        x0 = x0 - lr * gradiente
        iteracion += 1
    return x0, lista_normas, iteracion

def momentum(grad_f,x0,ruido,lam,lr,max_ite,eps,eta):
    lista_normas = []
    iteracion = 0
    v = np.zeros_like(x0)
    for i in range(max_ite):
        gradiente = np.array(grad_f(x0,ruido,lam))
        norma = np.linalg.norm(gradiente)
        lista_normas.append(norma)
        if norma < eps:
            break
        vi = eta*v + lr*gradiente
        xi = x0 - vi
        x0= xi.copy()
        v = vi.copy()
        iteracion += 1
    return x0, lista_normas, iteracion

def nesterov(grad_f, x0, ruido, lam, lr, max_ite, eps, eta):
    lista_normas = []
    iteracion = 0
    v = np.zeros_like(x0)
    for i in range(max_ite):
        x_adelantado = x0 - eta * v
        grad = grad_f(x_adelantado, ruido, lam)
        norma = np.linalg.norm(grad)
        lista_normas.append(norma)
        v = eta * v + lr * grad
        x0 = x0 - v
        
        iteracion += 1
        if norma < eps:
            break
    return x0, lista_normas, iteracion

def gradiente_J(x, ruido, lam):
    grad = x - ruido

    grad[:-1, :] += lam * (x[:-1, :] - x[1:, :])
    grad[1:, :]  += lam * (x[1:, :] - x[:-1, :])
    
    grad[:, :-1] += lam * (x[:, :-1] - x[:, 1:])
    grad[:, 1:]  += lam * (x[:, 1:] - x[:, :-1])
    
    return grad