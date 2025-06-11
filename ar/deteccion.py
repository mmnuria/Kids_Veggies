import cv2
import numpy as np

def crear_detector():
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    return cv2.aruco.ArucoDetector(diccionario)

def detectar_pose(frame, tam, detector, cameraMatrix, distCoeffs):
    bboxs, ids, _ = detector.detectMarkers(frame)
    print("ids: ", ids)
    if ids is not None:
        objPoints = np.array([[-tam/2.0, tam/2.0, 0.0],
                              [tam/2.0, tam/2.0, 0.0],
                              [tam/2.0, -tam/2.0, 0.0],
                              [-tam/2.0, -tam/2.0, 0.0]])
        resultado = {}
        ids = ids.flatten()
        for i in range(len(ids)):
            imagePoints = bboxs[i].reshape((4, 2)) 
            ret, rvec, tvec = cv2.solvePnP(objPoints, imagePoints, cameraMatrix, distCoeffs)
            if ret:
                resultado[ids[i]] = (rvec, tvec)
        return (True, resultado)
    return (False, None)
