import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
import time

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
            print(f"Convergió en la iteración {i}")
            break
    return x0, lista_normas, iteracion


max_iter = 1000
eps = 1e-6
lr = 0.002
lam = 0.1
eta=0.9

imagen_pred = np.random.rand(*imagen_ruido.shape)

inicio_1 = time.time()
imagen_pred, error_simple, iteracion_simple = descenso_grad(gradiente_J, imagen_pred, imagen_ruido,lam, lr, max_iter, eps)
fin_1 = time.time()
inicio_2 = time.time()
imagen_pred2, error_momentum, iteracion_momentum = momentum(gradiente_J, imagen_pred, imagen_ruido,lam, lr, max_iter, eps,eta)
fin_2 = time.time()
inicio_3 = time.time()
imagen_pred3, error_nesterov, iteracion_nesterov = nesterov(gradiente_J, imagen_pred, imagen_ruido,lam, lr, max_iter, eps,eta)
fin_3 = time.time()

print("\nIteraciones Descenso de Gradiente:", iteracion_simple,
      "Error:", error_simple[-1], "Tiempo:", fin_1 - inicio_1)
print("\nIteraciones Momentum:", iteracion_momentum,
      "Error:", error_momentum[-1], "Tiempo:", fin_2 - inicio_2)
print("\nIteraciones Nesterov:", iteracion_nesterov,
      "Error:", error_nesterov[-1], "Tiempo:", fin_3 - inicio_3,'\n')

fig, ax = plt.subplots(1, 4, figsize=(12, 6))
ax[0].imshow(io.imread('Ruido/imagenes/nutria_ruido.jpg'), cmap='gray')
ax[0].set_title('Imagen Original')
ax[0].axis('off')

ax[1].imshow(imagen_pred, cmap='gray')
ax[1].set_title('Predicción 1 (Descenso de Gradiente)')
ax[1].axis('off')

ax[2].imshow(imagen_pred2, cmap='gray')
ax[2].set_title('Predicción 2 (Momentum)')
ax[2].axis('off')

ax[3].imshow(imagen_pred3, cmap='gray')
ax[3].set_title('Predicción 3 (Nesterov)')
ax[3].axis('off')

plt.show()

plt.figure(figsize=(10, 5))
plt.title('Error de la imagen')
plt.plot(error_simple, label='Descenso de Gradiente')
plt.plot(error_momentum, label='Momentum')
plt.plot(error_nesterov, label='Nesterov')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.legend()
plt.show()