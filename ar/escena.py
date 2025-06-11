import cuia
import numpy as np

def fov(cameraMatrix, ancho, alto):
    if ancho > alto:
        f = cameraMatrix[1, 1]
        fov_rad = 2 * np.arctan(alto / (2 * f))
    else:
        f = cameraMatrix[0, 0]
        fov_rad = 2 * np.arctan(ancho / (2 * f))
    return np.rad2deg(fov_rad)

def crear_escena(modelo, cameraMatrix, ancho, alto):
    escena = cuia.escenaPYGFX(fov(cameraMatrix, ancho, alto), ancho, alto)
    escena.agregar_modelo(modelo)
    escena.ilumina_modelo(modelo)
    escena.iluminar()
    return escena
