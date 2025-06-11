import cv2
import numpy as np
import speech_recognition as sr
import threading
import time
import cuia
from config.calibracion import cargar_calibracion
from models.modelo_pera import crear_modelo
from ar.escena import crear_escena
from ar.deteccion import crear_detector, detectar_pose
from utils.conversiones import from_opencv_to_pygfx

# ----- Estados de la aplicación -----
class GameState:
    def __init__(self):
        self.fase = "reconocimiento_facial"
        self.usuario_identificado = False
        self.pregunta_realizada = False
        self.respuesta_recibida = ""
        self.respuesta_correcta = False
        self.tiempo_pregunta = 0
        self.mostrar_resultado = False
        self.tiempo_resultado = 0
        self.microfono_listo = False
        self.esperando_voz = False

# ----- Configuración de reconocimiento facial -----
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----- Variables globales -----
state = GameState()
voice_thread_active = False
recognizer = None
microphone = None

# ----- Función mejorada de reconocimiento de voz -----
def inicializar_microfono():
    """Inicializa el micrófono de forma no bloqueante"""
    global recognizer, microphone, state
    
    try:
        print("🎤 Inicializando micrófono...")
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        # Configuración rápida del micrófono (sin adjust_for_ambient_noise)
        recognizer.energy_threshold = 4000  # Ajusta según tu entorno
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.phrase_threshold = 0.3
        
        state.microfono_listo = True
        print("✅ Micrófono listo")
        
    except Exception as e:
        print(f"❌ Error configurando micrófono: {e}")
        state.microfono_listo = False

def reconocimiento_voz():
    """Función mejorada de reconocimiento de voz"""
    global state, voice_thread_active, recognizer, microphone
    
    voice_thread_active = True
    
    while voice_thread_active:
        # Solo procesar si estamos esperando respuesta y el micrófono está listo
        if state.esperando_voz and state.microfono_listo and recognizer and microphone:
            try:
                print("🎤 Escuchando...")
                with microphone as source:
                    # Escuchar con timeout más corto
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
                    
                print("🔄 Procesando audio...")
                texto = recognizer.recognize_google(audio, language="es-ES")
                texto = texto.lower().strip()
                print(f"🎤 Detectado: '{texto}'")
                
                # Verificar respuesta
                if "pera" in texto:
                    state.respuesta_recibida = "pera"
                    state.respuesta_correcta = True
                    print("✅ ¡Respuesta correcta!")
                else:
                    state.respuesta_recibida = texto
                    state.respuesta_correcta = False
                    print("❌ Respuesta incorrecta")
                
                # Cambiar estado
                state.fase = "resultado"
                state.esperando_voz = False
                state.mostrar_resultado = True
                state.tiempo_resultado = time.time()
                
            except sr.WaitTimeoutError:
                # Timeout normal, continuar
                continue
            except sr.UnknownValueError:
                print("🤷 No se entendió, intenta de nuevo...")
                continue
            except sr.RequestError as e:
                print(f"❌ Error del servicio: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"❌ Error inesperado en reconocimiento: {e}")
                time.sleep(1)
                continue
        else:
            time.sleep(0.1)

# ----- Función de realidad aumentada -----
def realidad_mixta(frame, escena, detector, cameraMatrix, distCoeffs):
    global state
    
    # Solo mostrar la pera si el usuario está identificado y estamos en fase de pregunta o después
    if state.usuario_identificado and state.fase in ["pregunta", "esperando_respuesta", "resultado"]:
        ret, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
        if ret and pose is not None and 0 in pose:
            M = from_opencv_to_pygfx(pose[0][0], pose[0][1])
            escena.actualizar_camara(M)
            imagen_render = escena.render()
            imagen_render_bgr = cv2.cvtColor(imagen_render, cv2.COLOR_RGBA2BGRA)
            resultado = cuia.alphaBlending(imagen_render_bgr, frame)
            return resultado
    
    return frame

# ----- Función para dibujar texto con fondo -----
def draw_text_with_background(img, text, pos, font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Obtener tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Dibujar rectángulo de fondo
    cv2.rectangle(img, 
                  (pos[0] - 5, pos[1] - text_height - 5),
                  (pos[0] + text_width + 5, pos[1] + baseline + 5),
                  bg_color, -1)
    
    # Dibujar texto
    cv2.putText(img, text, pos, font, font_scale, color, thickness)

# ----- Función principal -----
def main():
    global state, voice_thread_active
    
    cam = 0
    bk = cuia.bestBackend(cam)
    
    # Configurar cámara y parámetros AR
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
 
    # Inicializar micrófono en hilo separado
    hilo_microfono = threading.Thread(target=inicializar_microfono, daemon=True)
    hilo_microfono.start()

    # Iniciar hilo de reconocimiento de voz
    hilo_voz = threading.Thread(target=reconocimiento_voz, daemon=True)
    hilo_voz.start()

    print("🎮 Kids&Veggies iniciado - Mira a la cámara para comenzar")

    try:
        while True:
            ret, frame = ar.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_time = time.time()

            # ----- FASE 1: Reconocimiento Facial -----
            if state.fase == "reconocimiento_facial":
                faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    draw_text_with_background(frame, "¡Usuario identificado!", (x, y-15), 
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                    
                    state.usuario_identificado = True
                    state.fase = "pregunta"
                    state.tiempo_pregunta = current_time
                    print("👦 Usuario identificado - Iniciando juego...")
                else:
                    draw_text_with_background(frame, "Mira a la camara para comenzar", (50, 50),
                                            color=(255, 255, 255), bg_color=(100, 100, 100))
                
                # Mostrar estado del micrófono
                if state.microfono_listo:
                    draw_text_with_background(frame, "🎤 Microfono listo", (50, 100),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                else:
                    draw_text_with_background(frame, "🎤 Configurando microfono...", (50, 100),
                                            color=(255, 255, 0), bg_color=(100, 100, 0))

            # ----- FASE 2: Mostrar pregunta -----
            elif state.fase == "pregunta":
                draw_text_with_background(frame, "Coloca el marcador ArUco frente a la camara", (50, 50),
                                        color=(255, 255, 0), bg_color=(100, 100, 0))
                # Verificar si hay marcador visible
                ret_pose, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
                print(ret_pose, pose)
                if ret_pose and pose is not None:  # Simplificado y correcto
                    if not state.pregunta_realizada:
                        draw_text_with_background(frame, "¿Que fruta es esta?", (50, 100), color=(0, 255, 255), bg_color=(100, 0, 100))
                        draw_text_with_background(frame, "Di el nombre en voz alta", (50, 150), color=(255, 255, 255), bg_color=(50, 50, 50))
                        
                        # Cambiar a fase de espera después de 3 segundos
                        if current_time - state.tiempo_pregunta > 3:
                            if state.microfono_listo:
                                state.fase = "esperando_respuesta"
                                state.pregunta_realizada = True
                                state.esperando_voz = True
                                print("❓ Pregunta realizada - Esperando respuesta de voz...")
                            else:
                                draw_text_with_background(frame, "Esperando microfono...", (50, 200), color=(255, 255, 0), bg_color=(100, 100, 0))
            # ----- FASE 3: Esperando respuesta -----
            elif state.fase == "esperando_respuesta":
                draw_text_with_background(frame, "Escuchando tu respuesta...", (50, 50),
                                        color=(255, 0, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "🎤 Habla ahora", (50, 100),
                                        color=(255, 255, 255), bg_color=(50, 50, 50))

            # ----- FASE 4: Mostrar resultado -----
            elif state.fase == "resultado":
                if state.respuesta_correcta:
                    draw_text_with_background(frame, "¡CORRECTO! Es una pera", (50, 50),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                    draw_text_with_background(frame, "¡Muy bien!", (50, 100),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                else:
                    draw_text_with_background(frame, f"No es correcto. Dijiste: {state.respuesta_recibida}", (50, 50),
                                            color=(0, 0, 255), bg_color=(100, 0, 0))
                    draw_text_with_background(frame, "La respuesta correcta es: PERA", (50, 100),
                                            color=(255, 255, 0), bg_color=(100, 100, 0))
                
                # Reiniciar después de 5 segundos
                if current_time - state.tiempo_resultado > 5:
                    print("🔄 Reiniciando juego...")
                    state = GameState()
                    # Reinicializar micrófono si es necesario
                    if not state.microfono_listo:
                        hilo_microfono = threading.Thread(target=inicializar_microfono, daemon=True)
                        hilo_microfono.start()

            # Mostrar instrucciones de salida
            draw_text_with_background(frame, "Presiona ESC para salir", (ancho-250, alto-30),
                                    font_scale=0.5, color=(200, 200, 200), bg_color=(50, 50, 50))

            cv2.imshow("Kids&Veggies - AR Learning Game", frame)

            # Salir con ESC
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("\n🛑 Aplicación interrumpida por el usuario")
    
    finally:
        voice_thread_active = False
        ar.release()
        cv2.destroyAllWindows()
        print("👋 ¡Hasta luego!")

if __name__ == "__main__":
    main()