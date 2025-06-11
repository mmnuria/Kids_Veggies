import cv2
import numpy as np
import speech_recognition as sr
import threading
import time
import cuia
from config.calibracion import cargar_calibracion
from models.modelos import MODELOS_FRUTAS_VERDURAS, crear_modelo_por_id, obtener_info_modelo
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
        self.marker_id_actual = None
        self.modelo_actual = None
        self.info_modelo_actual = None

# ----- Configuración de reconocimiento facial -----
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ----- Variables globales -----
state = GameState()
voice_thread_active = False
recognizer = None
microphone = None
escenas = {}  # Diccionario para almacenar las escenas de cada modelo

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

def verificar_respuesta(texto, respuesta_correcta):
    """Verificar si la respuesta del usuario es correcta"""
    texto = texto.lower().strip()
    respuesta_correcta = respuesta_correcta.lower().strip()
    
    # Verificaciones directas
    if respuesta_correcta in texto:
        return True
    
    # Verificaciones adicionales para variaciones comunes
    variaciones = {
        'pera': ['pera'],
        'cebolleta': ['cebolleta', 'cebolla verde', 'cebollín'],
        'cebolla': ['cebolla'],
        'lechuga': ['lechuga'],
        'limon': ['limón', 'limon'],
        'pimiento rojo': ['pimiento rojo', 'pimiento', 'chile rojo'],
        'pimiento verde': ['pimiento verde', 'pimiento', 'chile verde'],
        'uvas': ['uvas', 'uva'],
        'zanahoria': ['zanahoria']
    }
    
    if respuesta_correcta in variaciones:
        for variacion in variaciones[respuesta_correcta]:
            if variacion in texto:
                return True
    
    return False

def reconocimiento_voz():
    """Función mejorada de reconocimiento de voz"""
    global state, voice_thread_active, recognizer, microphone
    
    voice_thread_active = True
    
    while voice_thread_active:
        # Solo procesar si estamos esperando respuesta y el micrófono está listo
        if state.esperando_voz and state.microfono_listo and recognizer and microphone and state.info_modelo_actual:
            try:
                print("🎤 Escuchando...")
                with microphone as source:
                    # Escuchar con timeout más corto
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
                    
                print("🔄 Procesando audio...")
                texto = recognizer.recognize_google(audio, language="es-ES")
                texto = texto.lower().strip()
                print(f"🎤 Detectado: '{texto}'")
                
                # Verificar respuesta usando la función mejorada
                respuesta_correcta = state.info_modelo_actual['respuesta_correcta']
                if verificar_respuesta(texto, respuesta_correcta):
                    state.respuesta_recibida = texto
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

# ----- Función de realidad aumentada mejorada -----
def realidad_mixta(frame, detector, cameraMatrix, distCoeffs):
    global state, escenas
    
    # Solo mostrar el modelo si el usuario está identificado y estamos en fase de pregunta o después
    if state.usuario_identificado and state.fase in ["pregunta", "esperando_respuesta", "resultado"]:
        ret, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
        
        if ret and pose is not None:
            # Obtener todos los IDs de marcadores detectados
            marker_ids = list(pose.keys())
            
            if marker_ids:
                # Usar el primer marcador detectado
                marker_id = marker_ids[0]
                
                # Si es un nuevo marcador, crear/cambiar el modelo
                if state.marker_id_actual != marker_id:
                    state.marker_id_actual = marker_id
                    state.info_modelo_actual = obtener_info_modelo(marker_id)
                    
                    # Crear escena para este modelo si no existe
                    if marker_id not in escenas:
                        modelo = crear_modelo_por_id(marker_id)
                        escenas[marker_id] = crear_escena(modelo, cameraMatrix, 
                                                        int(frame.shape[1]), int(frame.shape[0]))
                    
                    print(f"🍎 Mostrando: {state.info_modelo_actual['nombre']} (ID: {marker_id})")
                
                # Renderizar el modelo actual
                if state.marker_id_actual in escenas:
                    M = from_opencv_to_pygfx(pose[marker_id][0], pose[marker_id][1])
                    escenas[state.marker_id_actual].actualizar_camara(M)
                    imagen_render = escenas[state.marker_id_actual].render()
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
    global state, voice_thread_active, escenas
    
    cam = 0
    bk = cuia.bestBackend(cam)
    
    # Configurar cámara y parámetros AR
    webcam = cv2.VideoCapture(cam, bk)
    ancho = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    webcam.release()
    
    cameraMatrix, distCoeffs = cargar_calibracion(ancho, alto)
    detector = crear_detector()

    ar = cuia.myVideo(cam, bk)
    ar.process = lambda frame: realidad_mixta(frame, detector, cameraMatrix, distCoeffs)
 
    # Inicializar micrófono en hilo separado
    hilo_microfono = threading.Thread(target=inicializar_microfono, daemon=True)
    hilo_microfono.start()

    # Iniciar hilo de reconocimiento de voz
    hilo_voz = threading.Thread(target=reconocimiento_voz, daemon=True)
    hilo_voz.start()

    print("🎮 Kids&Veggies iniciado - Mira a la cámara para comenzar")
    print("📱 Marcadores disponibles:")
    for marker_id, info in MODELOS_FRUTAS_VERDURAS.items():
        print(f"   ID {marker_id}: {info['nombre']} ({info['tipo']})")

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
                draw_text_with_background(frame, "Coloca un marcador ArUco frente a la camara", (50, 50),
                                        color=(255, 255, 0), bg_color=(100, 100, 0))
                
                # Mostrar qué marcador está siendo detectado
                if state.marker_id_actual is not None and state.info_modelo_actual:
                    draw_text_with_background(frame, f"Detectado: {state.info_modelo_actual['nombre']}", (50, 100),
                                            color=(255, 255, 255), bg_color=(50, 50, 50))
                
                # Verificar si hay marcador visible
                ret_pose, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
                if ret_pose and pose is not None and state.info_modelo_actual:
                    if not state.pregunta_realizada:
                        tipo = state.info_modelo_actual['tipo']
                        draw_text_with_background(frame, f"¿Que {tipo} es esta?", (50, 150), 
                                                color=(0, 255, 255), bg_color=(100, 0, 100))
                        draw_text_with_background(frame, "Di el nombre en voz alta", (50, 200), 
                                                color=(255, 255, 255), bg_color=(50, 50, 50))
                        
                        # Cambiar a fase de espera después de 3 segundos
                        if current_time - state.tiempo_pregunta > 3:
                            if state.microfono_listo:
                                state.fase = "esperando_respuesta"
                                state.pregunta_realizada = True
                                state.esperando_voz = True
                                print(f"❓ Pregunta realizada sobre {state.info_modelo_actual['nombre']} - Esperando respuesta de voz...")
                            else:
                                draw_text_with_background(frame, "Esperando microfono...", (50, 250), 
                                                        color=(255, 255, 0), bg_color=(100, 100, 0))

            # ----- FASE 3: Esperando respuesta -----
            elif state.fase == "esperando_respuesta":
                if state.info_modelo_actual:
                    draw_text_with_background(frame, f"¿Que {state.info_modelo_actual['tipo']} es esta?", (50, 50),
                                            color=(0, 255, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "Escuchando tu respuesta...", (50, 100),
                                        color=(255, 0, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "🎤 Habla ahora", (50, 150),
                                        color=(255, 255, 255), bg_color=(50, 50, 50))

            # ----- FASE 4: Mostrar resultado -----
            elif state.fase == "resultado":
                if state.respuesta_correcta and state.info_modelo_actual:
                    draw_text_with_background(frame, f"¡CORRECTO! Es {state.info_modelo_actual['nombre']}", (50, 50),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                    draw_text_with_background(frame, "¡Muy bien!", (50, 100),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                else:
                    if state.info_modelo_actual:
                        draw_text_with_background(frame, f"No es correcto. Dijiste: {state.respuesta_recibida}", (50, 50),
                                                color=(0, 0, 255), bg_color=(100, 0, 0))
                        draw_text_with_background(frame, f"La respuesta correcta es: {state.info_modelo_actual['nombre'].upper()}", (50, 100),
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