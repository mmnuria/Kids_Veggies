import numpy as np

def cargar_calibracion(ancho, alto):
    try:
        import config.camara as camara
        return camara.cameraMatrix, camara.distCoeffs
    except ImportError:
        cameraMatrix = np.array([[1000, 0, ancho / 2],
                                 [0, 1000, alto / 2],
                                 [0, 0, 1]])
        distCoeffs = np.zeros((5, 1))
        return cameraMatrix, distCoeffs
