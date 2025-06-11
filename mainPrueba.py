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
        
        # Variables del sistema de progreso
        self.marcadores_detectados = set()  # IDs de marcadores que se han detectado
        self.marcadores_respondidos = set()  # IDs de marcadores que se han respondido correctamente
        self.marcadores_pendientes = set()  # IDs de marcadores que faltan por responder
        self.puntuacion = 0
        self.total_preguntas = 0
        self.juego_completado = False
        self.tiempo_escaneo = 0
        self.en_fase_escaneo = False

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
    #Inicializa el micrófono de forma no bloqueante
    global recognizer, microphone, state
    
    try:
        print("🎤 Inicializando micrófono...")
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        # Configuración micrófono
        recognizer.energy_threshold = 4000
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.phrase_threshold = 0.3
        
        state.microfono_listo = True
        print("***** Micrófono listo *****")
        
    except Exception as e:
        print(f"xxxxxx Error configurando micrófono: {e} xxxxxxx")
        state.microfono_listo = False

def verificar_respuesta(texto, respuesta_correcta):
    #Verificar si la respuesta del usuario es correcta
    texto = texto.lower().strip()
    respuesta_correcta = respuesta_correcta.lower().strip()
    
    # Verificaciones directas
    if respuesta_correcta in texto:
        return True
    
    # Verificaciones adicionales
    variaciones = {
        'pera': ['pera'],
        'cebolleta': ['cebolleta'],
        'cebolla': ['cebolla'],
        'lechuga': ['lechuga'],
        'limon': ['limón', 'limon'],
        'pimiento rojo': ['pimiento rojo', 'pimiento'],
        'pimiento verde': ['pimiento verde', 'pimiento'],
        'uvas': ['uvas', 'uva'],
        'zanahoria': ['zanahoria']
    }
    
    if respuesta_correcta in variaciones:
        for variacion in variaciones[respuesta_correcta]:
            if variacion in texto:
                return True
    
    return False

def reconocimiento_voz():
    global state, voice_thread_active, recognizer, microphone
    
    voice_thread_active = True
    
    while voice_thread_active:
        if state.esperando_voz and state.microfono_listo and recognizer and microphone and state.info_modelo_actual:
            try:
                print("***** Escuchando... *****")
                with microphone as source:
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
                    
                print("***** Procesando audio... *****")
                texto = recognizer.recognize_google(audio, language="es-ES")
                texto = texto.lower().strip()
                print(f"🎤 Detectado: '{texto}'")
                
                # Verificar respuesta
                respuesta_correcta = state.info_modelo_actual['respuesta_correcta']
                if verificar_respuesta(texto, respuesta_correcta):
                    state.respuesta_recibida = texto
                    state.respuesta_correcta = True
                    state.puntuacion += 1
                    state.marcadores_respondidos.add(state.marker_id_actual)
                    state.marcadores_pendientes.discard(state.marker_id_actual)
                    print(f"✅ ¡Respuesta correcta! Puntuación: {state.puntuacion}")
                else:
                    state.respuesta_recibida = texto
                    state.respuesta_correcta = False
                    print("❌ Respuesta incorrecta")
                
                # Actualizar total de preguntas
                state.total_preguntas += 1
                
                # Cambiar estado
                state.fase = "resultado"
                state.esperando_voz = False
                state.mostrar_resultado = True
                state.tiempo_resultado = time.time()
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("***** No se entendió, intenta de nuevo... *****")
                continue
            except sr.RequestError as e:
                print(f"xxxxx Error del servicio: {e} xxxxx")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"xxxxx Error inesperado en reconocimiento: {e} xxxxx")
                time.sleep(1)
                continue
        else:
            time.sleep(0.1)

# ----- Función para detectar marcadores disponibles -----
def detectar_marcadores_disponibles(frame, detector, cameraMatrix, distCoeffs):
    ret, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
    marcadores_encontrados = set()
    
    if ret and pose is not None:
        for marker_id in pose.keys():
            if marker_id in MODELOS_FRUTAS_VERDURAS:
                marcadores_encontrados.add(marker_id)
    
    return marcadores_encontrados

# ----- Función de realidad aumentada -----
def realidad_mixta(frame, detector, cameraMatrix, distCoeffs):
    global state, escenas

    marcadores_actuales = detectar_marcadores_disponibles(frame, detector, cameraMatrix, distCoeffs)

    if state.fase == "escaneo_inicial":
        state.marcadores_detectados.update(marcadores_actuales)

    if state.usuario_identificado and state.fase in ["pregunta", "esperando_respuesta", "resultado"]:
        ret, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)

        if ret and pose:
            marker_ids = list(pose.keys())
            if not marker_ids:
                return frame

            # 🔍 Seleccionar el marcador prioritario (pendiente si lo hay)
            marker_id = next((mid for mid in marker_ids if mid in state.marcadores_pendientes), marker_ids[0])

            # 🛠️ Asegurar que la escena esté creada
            if marker_id not in escenas:
                print(f"🔧 Creando nueva escena para marcador {marker_id}")
                modelo = crear_modelo_por_id(marker_id)
                if modelo is None:
                    print(f"⚠️ No se pudo crear modelo para el marcador {marker_id}")
                    return frame
                escenas[marker_id] = crear_escena(modelo, cameraMatrix,
                                                  int(frame.shape[1]), int(frame.shape[0]))

            # 🧠 Actualizar estado si cambia el marcador
            if state.marker_id_actual != marker_id:
                print(f"🆕 Cambiando marcador activo: {state.marker_id_actual} → {marker_id}")
                state.marker_id_actual = marker_id
                state.info_modelo_actual = obtener_info_modelo(marker_id)
                print(f"🍎 Mostrando: {state.info_modelo_actual['nombre']} (ID: {marker_id})")

            # 🎬 Renderizar escena
            escena = escenas.get(marker_id)
            if escena:
                rvec, tvec = pose[marker_id]
                M = from_opencv_to_pygfx(rvec, tvec)
                escena.actualizar_camara(M)
                imagen_render = escena.render()
                imagen_render_bgr = cv2.cvtColor(imagen_render, cv2.COLOR_RGBA2BGRA)
                return cuia.alphaBlending(imagen_render_bgr, frame)

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

# ----- Función para mostrar progreso -----
def mostrar_progreso(frame):
    """Muestra el progreso del juego"""
    y_pos = 50
    
    # Mostrar puntuación
    draw_text_with_background(frame, f"Puntuación: {state.puntuacion}/{len(state.marcadores_detectados)}", 
                            (50, y_pos), color=(255, 255, 0), bg_color=(100, 100, 0))
    y_pos += 40
    
    # Mostrar marcadores detectados
    if state.marcadores_detectados:
        draw_text_with_background(frame, f"Marcadores encontrados: {len(state.marcadores_detectados)}", 
                                (50, y_pos), color=(0, 255, 255), bg_color=(100, 100, 0))
        y_pos += 30
        
        # Mostrar estado de cada marcador
        for marker_id in sorted(state.marcadores_detectados):
            info = obtener_info_modelo(marker_id)
            if marker_id in state.marcadores_respondidos:
                estado = "✅"
                color = (0, 255, 0)
                bg_color = (0, 100, 0)
            else:
                estado = "❓"
                color = (255, 255, 0)
                bg_color = (100, 100, 0)
            
            draw_text_with_background(frame, f"{estado} {info['nombre']}", 
                                    (70, y_pos), font_scale=0.6, color=color, bg_color=bg_color)
            y_pos += 25

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
            if not ret or frame is None:
                print("⚠️ Frame no válido")
                continue  # o manejar de otra forma

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
                    state.fase = "escaneo_inicial"
                    state.tiempo_escaneo = current_time
                    state.en_fase_escaneo = True
                    print("👦 Usuario identificado - Buscando marcadores...")
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

            # ----- FASE NUEVA: Escaneo inicial de marcadores -----
            elif state.fase == "escaneo_inicial":
                draw_text_with_background(frame, "Muestra todos los marcadores ArUco", (50, 50),
                                        color=(255, 255, 0), bg_color=(100, 100, 0))
                draw_text_with_background(frame, "que quieres incluir en el juego", (50, 100),
                                        color=(255, 255, 0), bg_color=(100, 100, 0))
                
                # Mostrar marcadores detectados hasta ahora
                if state.marcadores_detectados:
                    draw_text_with_background(frame, f"Marcadores encontrados: {len(state.marcadores_detectados)}", (50, 150),
                                            color=(0, 255, 255), bg_color=(0, 100, 100))
                    
                    y_pos = 180
                    for marker_id in sorted(state.marcadores_detectados):
                        info = obtener_info_modelo(marker_id)
                        draw_text_with_background(frame, f"• {info['nombre']}", (70, y_pos),
                                                font_scale=0.6, color=(255, 255, 255), bg_color=(50, 50, 50))
                        y_pos += 25
                
                # Después de 10 segundos de escaneo, comenzar el juego
                tiempo_transcurrido = current_time - state.tiempo_escaneo
                tiempo_restante = max(0, 10 - int(tiempo_transcurrido))
                
                draw_text_with_background(frame, f"Tiempo restante: {tiempo_restante}s", (50, alto-80),
                                        color=(255, 255, 255), bg_color=(100, 100, 100))
                
                # Después de 10 segundos desde el inicio del escaneo...
                if tiempo_transcurrido > 10:
                    if state.marcadores_detectados:
                        # Si se detectaron marcadores, avanzar al juego
                        state.marcadores_pendientes = state.marcadores_detectados.copy()
                        state.fase = "pregunta"
                        state.tiempo_pregunta = current_time
                        state.en_fase_escaneo = False
                        print(f"🎯 Comenzando juego con {len(state.marcadores_detectados)} marcadores")
                    else:
                        # Si no hay marcadores aún, mostrar advertencia
                        draw_text_with_background(frame, "No se encontraron marcadores", (50, alto-50),
                                                color=(255, 0, 0), bg_color=(100, 0, 0))
                        draw_text_with_background(frame, "Asegúrate de mostrar al menos uno", (50, alto-20),
                                                color=(255, 255, 255), bg_color=(100, 100, 100))
                        
                        # Si pasaron más de 15 segundos en total, reinicia el tiempo de escaneo
                        if tiempo_transcurrido > 15:
                            print("🔁 Reiniciando escaneo de marcadores...")
                            state.tiempo_escaneo = current_time  # Reinicia el tiempo


            # ----- FASE 2: Mostrar pregunta -----
            elif state.fase == "pregunta":
                # Verificar si hay marcadores pendientes
                if not state.marcadores_pendientes:
                    state.fase = "juego_completado"
                    state.juego_completado = True
                    print("🎉 ¡Juego completado!")
                else:
                    # Mostrar progreso
                    mostrar_progreso(frame)
                    
                    draw_text_with_background(frame, "Coloca un marcador pendiente", (50, alto-150),
                                            color=(255, 255, 0), bg_color=(100, 100, 0))
                    
                    # Mostrar qué marcador está siendo detectado
                    if state.marker_id_actual is not None and state.info_modelo_actual:
                        if state.marker_id_actual in state.marcadores_detectados:
                            if state.marker_id_actual in state.marcadores_respondidos:
                                draw_text_with_background(frame, f"Ya respondido: {state.info_modelo_actual['nombre']}", (50, alto-120),
                                                        color=(100, 100, 100), bg_color=(50, 50, 50))
                            elif state.marker_id_actual in state.marcadores_pendientes:
                                draw_text_with_background(frame, f"Detectado: {state.info_modelo_actual['nombre']}", (50, alto-120),
                                                        color=(0, 255, 0), bg_color=(0, 100, 0))
                            else:
                                # No debería ocurrir, pero lo dejamos como seguridad
                                draw_text_with_background(frame, f"Marcador desconocido: {state.info_modelo_actual['nombre']}", (50, alto-120),
                                                        color=(255, 255, 0), bg_color=(100, 100, 0))
                        else:
                            # Marcador no era parte del juego original
                            draw_text_with_background(frame, f"Marcador NO incluido: {state.info_modelo_actual['nombre']}", (50, alto-120),
                                                    color=(0, 0, 255), bg_color=(100, 0, 0))

                    # Verificar si hay marcador pendiente visible y estable
                    ret_pose, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
                    marcador_valido = (
                        ret_pose and pose is not None
                        and state.marker_id_actual in state.marcadores_pendientes
                        and state.info_modelo_actual is not None
                    )

                    if marcador_valido:
                        tipo = state.info_modelo_actual['tipo']
                        draw_text_with_background(frame, f"¿Qué {tipo} es esta?", (50, alto-90), 
                                                color=(0, 255, 255), bg_color=(100, 0, 100))
                        draw_text_with_background(frame, "Di el nombre en voz alta", (50, alto-60), 
                                                color=(255, 255, 255), bg_color=(50, 50, 50))
                        print("🟢 Marcador válido y visible, esperando estabilidad para lanzar pregunta...")
                        # Esperar 2-3 segundos de estabilidad antes de preguntar
                        if not state.pregunta_realizada:
                            if current_time - state.tiempo_pregunta > 2:
                                if state.microfono_listo:
                                    state.fase = "esperando_respuesta"
                                    state.pregunta_realizada = True
                                    state.esperando_voz = True
                                    print(f"❓ Pregunta lanzada: {state.info_modelo_actual['nombre']} (ID {state.marker_id_actual})")
                                else:
                                    draw_text_with_background(frame, "Esperando micrófono...", (50, alto-30), 
                                                            color=(255, 255, 0), bg_color=(100, 100, 0))
                        else:
                            state.tiempo_pregunta = current_time



            # ----- FASE 3: Esperando respuesta -----
            elif state.fase == "esperando_respuesta":
                mostrar_progreso(frame)
                
                if state.info_modelo_actual:
                    draw_text_with_background(frame, f"¿Que {state.info_modelo_actual['tipo']} es esta?", (50, alto-120),
                                            color=(0, 255, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "Escuchando tu respuesta...", (50, alto-90),
                                        color=(255, 0, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "🎤 Habla ahora", (50, alto-60),
                                        color=(255, 255, 255), bg_color=(50, 50, 50))

            # ----- FASE 4: Mostrar resultado -----
            elif state.fase == "resultado":
                mostrar_progreso(frame)
                
                if state.respuesta_correcta and state.info_modelo_actual:
                    draw_text_with_background(frame, f"¡CORRECTO! Es {state.info_modelo_actual['nombre']}", (50, alto-120),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                    draw_text_with_background(frame, "¡Muy bien!", (50, alto-90),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                else:
                    if state.info_modelo_actual:
                        draw_text_with_background(frame, f"No es correcto. Dijiste: {state.respuesta_recibida}", (50, alto-120),
                                                color=(0, 0, 255), bg_color=(100, 0, 0))
                        draw_text_with_background(frame, f"La respuesta correcta es: {state.info_modelo_actual['nombre'].upper()}", (50, alto-90),
                                                color=(255, 255, 0), bg_color=(100, 100, 0))
                
                # Continuar después de 3 segundos
                if current_time - state.tiempo_resultado > 3:
                    if state.marcadores_pendientes:
                        state.fase = "pregunta"
                        state.pregunta_realizada = False
                        state.tiempo_pregunta = current_time
                        print("🔄 Continuando con siguiente pregunta...")
                    else:
                        state.fase = "juego_completado"
                        state.juego_completado = True
                        print("🎉 ¡Juego completado!")

            # ----- FASE 5: Juego completado -----
            elif state.fase == "juego_completado":
                # Mostrar estadísticas finales
                draw_text_with_background(frame, "¡FELICIDADES!", (ancho//2-100, 100),
                                        font_scale=1.2, color=(0, 255, 0), bg_color=(0, 100, 0))
                
                draw_text_with_background(frame, "Has completado el juego", (ancho//2-120, 150),
                                        color=(255, 255, 0), bg_color=(100, 100, 0))
                
                porcentaje = int((state.puntuacion / len(state.marcadores_detectados)) * 100) if state.marcadores_detectados else 0
                draw_text_with_background(frame, f"Puntuación final: {state.puntuacion}/{len(state.marcadores_detectados)} ({porcentaje}%)", 
                                        (ancho//2-150, 200), color=(255, 255, 255), bg_color=(50, 50, 50))
                
                # Mostrar todas las frutas/verduras identificadas
                y_pos = 250
                draw_text_with_background(frame, "Frutas y verduras identificadas:", (50, y_pos),
                                        color=(0, 255, 255), bg_color=(0, 100, 100))
                y_pos += 40
                
                for marker_id in sorted(state.marcadores_detectados):
                    info = obtener_info_modelo(marker_id)
                    if marker_id in state.marcadores_respondidos:
                        estado = "✅"
                        color = (0, 255, 0)
                    else:
                        estado = "❌"
                        color = (255, 0, 0)
                    
                    draw_text_with_background(frame, f"{estado} {info['nombre']} ({info['tipo']})", 
                                            (70, y_pos), font_scale=0.6, color=color, bg_color=(50, 50, 50))
                    y_pos += 25
                
                draw_text_with_background(frame, "Presiona ESC para salir", (ancho//2-100, alto-50),
                                        color=(255, 255, 255), bg_color=(100, 100, 100))

            # Mostrar instrucciones de salida
            if not state.juego_completado:
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