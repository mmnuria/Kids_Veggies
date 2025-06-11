import cv2
import numpy as np

# Configurar detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar marcadores
    corners, ids, rejected = detector.detectMarkers(gray)
    
    print(f"IDs detectados: {ids}")
    
    # Dibujar marcadores detectados
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"¡Detectados {len(ids)} marcadores!")
    
    cv2.imshow('Test ArUco', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()