import cv2
import numpy as np
import speech_recognition as sr
import threading
from cuia import myVideo, bestBackend, popup
from camara import cameraMatrix as K, distCoeffs as dist


# ----- Configuración de marcadores ArUco -----
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, PARAMS)

# Asociación ID de marcador -> fruta
FRUTA_POR_ID = {
    0: "Manzana",
    1: "Plátano",
    2: "Fresa",
    3: "Zanahoria",
    4: "Tomate",
    18: "Limón",
    29: "Cereza",
    37: "Naranja",
    48: "Brócoli"
}

# ----- Cargar clasificador Haar para rostro -----
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----- Función de reconocimiento de voz -----
def reconocimiento_voz(frutas_conocidas):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Esperando que digas una fruta...")

    while True:
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=5)
                texto = recognizer.recognize_google(audio, language="es-ES")
                texto = texto.lower().strip()
                print(f"🎤 Has dicho: {texto}")

                for fruta in frutas_conocidas.values():
                    if fruta.lower() in texto:
                        print(f"✅ Coincidencia con fruta: {fruta}")
                        break
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("🤷 No entendí lo que dijiste.")
            except sr.RequestError:
                print("❌ Error al conectar con el servicio de reconocimiento.")

# ----- Función principal -----
def main():
    backend = bestBackend(0)
    cam = myVideo(0, backend)

    if not cam.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    print("✅ Cámara inicializada correctamente.")
    identificado = False

    # Iniciar el hilo de voz
    hilo_voz = threading.Thread(target=reconocimiento_voz, args=(FRUTA_POR_ID,), daemon=True)
    hilo_voz.start()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----------- Fase 1: Reconocimiento Facial -----------
        if not identificado:
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame, "Usuario identificado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                identificado = True
                print("👦 Usuario identificado por rostro.")
        else:
            # ----------- Fase 2: Detección de ArUco -----------
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    fruta = FRUTA_POR_ID.get(marker_id, "Desconocida")
                    corner = corners[i][0]
                    center = corner.mean(axis=0).astype(int)
                    cv2.putText(frame, fruta, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    print(f"🍓 Marcador detectado (ID: {marker_id}) → {fruta}")
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Mostrar la ventana
        cv2.imshow("Kids&Veggies - Demo", frame)

        # Salir con tecla ESC
        if cv2.waitKey(20) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# ----- Ejecutar -----
if __name__ == "__main__":
    main()
