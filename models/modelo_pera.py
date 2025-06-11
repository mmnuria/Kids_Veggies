import cuia
import numpy as np

def crear_modelo():
    modelo = cuia.modeloGLTF('media/pera.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))  # Igual que la Tierra, si lo hiciste en Blender
    modelo.escalar(0.15)              # Ajusta esto según el tamaño de tu pera.glb
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo
