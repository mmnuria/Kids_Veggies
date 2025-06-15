import cv2
import numpy as np

def crear_detector():
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    return cv2.aruco.ArucoDetector(diccionario)

def detectar_pose(frame, tam, detector, cameraMatrix, distCoeffs):
    bboxs, ids, _ = detector.detectMarkers(frame)
    #print("ids: ", ids)
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

def ocultar_marcadores_visualmente(frame, detector):
    bboxs, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            pts = bboxs[i].astype(int)[0]

            x, y, w, h = cv2.boundingRect(pts)
            # Expandimos el área alrededor del marcador
            margin = 10
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, frame.shape[1])
            y2 = min(y + h + margin, frame.shape[0])

            # Crea una máscara vacía (todo ceros) del tamaño del rectángulo que rodea el marcador
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            #coordenadas absolutas (en el folio) de los vértices del marcador.
            pts_local = pts - [x1, y1]
            #Dibuja (rellena) el polígono del marcador en la máscara con valor 255 (blanco).
            cv2.fillPoly(mask, [pts_local], 255)
            #Recorta el frame para obtener sólo la región que rodea al marcador.
            region = frame[y1:y2, x1:x2]
            #Selecciona todos los píxeles que no están dentro del marcador (donde la máscara es 0 = negro).
            background_pixels = region[mask == 0]
            
            #Si hay píxeles en el fondo, calcula el color promedio en BGR de esos píxeles (promedio por canal).
            if len(background_pixels) > 0:
                avg_color_np = np.mean(background_pixels, axis=0)
                avg_color = tuple(int(c) for c in avg_color_np)
            else:
                avg_color = (0, 0, 0)

            # Dibujamos el polígono resultante sobre el marcador con el color promedio
            cv2.fillPoly(frame, [pts], color=avg_color)