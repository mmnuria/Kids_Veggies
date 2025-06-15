import numpy as np
import cv2

def from_opencv_to_pygfx(rvec, tvec):
    pose = np.eye(4)
    pose[0:3,3] = tvec.T
    pose[0:3,0:3] = cv2.Rodrigues(rvec)[0]
    pose[1:3] *= -1
    return np.linalg.inv(pose)

def mezclar_con_alpha(fondo_bgr, overlay_bgra):
    """
    Combina una imagen BGRA (con canal alfa) sobre un fondo BGR usando alpha blending.
    Ambos deben tener las mismas dimensiones.
    """
    if overlay_bgra.shape[2] != 4:
        raise ValueError("La imagen overlay debe tener 4 canales (BGRA)")

    # Separar componentes
    overlay_rgb = overlay_bgra[:, :, :3].astype(float)
    # Normalizar alpha a [0,1]
    alpha = overlay_bgra[:, :, 3].astype(float) / 255.0  
    # Para hacer broadcasting
    alpha = np.expand_dims(alpha, axis=2)  

    fondo = fondo_bgr.astype(float)

    # Mezcla: out = overlay * alpha + fondo * (1 - alpha)
    resultado = overlay_rgb * alpha + fondo * (1 - alpha)
    return resultado.astype(np.uint8)
