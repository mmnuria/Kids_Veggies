import cv2
import cuia
from config.calibracion import cargar_calibracion
from models.modelo_pera import crear_modelo
from ar.escena import crear_escena
from ar.deteccion import crear_detector, detectar_pose
from utils.conversiones import from_opencv_to_pygfx

def realidad_mixta(frame, escena, detector, cameraMatrix, distCoeffs):
    ret, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
    if ret and pose is not None and 0 in pose:
        M = from_opencv_to_pygfx(pose[0][0], pose[0][1])
        escena.actualizar_camara(M)
        imagen_render = escena.render()
        imagen_render_bgr = cv2.cvtColor(imagen_render, cv2.COLOR_RGBA2BGRA)
        resultado = cuia.alphaBlending(imagen_render_bgr, frame)
    else:
        resultado = frame
        
    return resultado

def main():
    cam = 0
    bk = cuia.bestBackend(cam)
    
    webcam = cv2.VideoCapture(cam, bk)
    ancho = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    webcam.release()
    
    cameraMatrix, distCoeffs = cargar_calibracion(ancho, alto)
    modelo = crear_modelo()
    escena = crear_escena(modelo, cameraMatrix, ancho, alto)
    detector = crear_detector()

    ar = cuia.myVideo(cam, bk)
    ar.process = lambda frame: realidad_mixta(frame, escena, detector, cameraMatrix, distCoeffs)

    try:
        ar.play("AR", key=27)
    finally:
        ar.release()

if __name__ == "__main__":
    main()