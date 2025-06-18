import cv2
import speech_recognition as sr
import threading
import time
import modules.cuia as cuia
import face_recognition
from config.calibracion import cargar_calibracion
from models.modelos import MODELOS_FRUTAS_VERDURAS, crear_modelo_por_id, obtener_info_modelo
from ar.escena import crear_escena
from ar.deteccion import crear_detector, detectar_pose, ocultar_marcadores_visualmente
from utils.conversiones import from_opencv_to_pygfx
from modules.usuarios import buscar_usuario_por_cara, guardar_puntuacion_juego, obtener_progreso_usuario, registrar_usuario, obtener_datos_visibles_usuario, verificar_usuario_existe, actualizar_nombre_usuario, actualizar_idioma_usuario
from modules.juegos import GestorJuegosAR, JuegoDescubreAR, JuegoEncuentraFrutasAR, JuegoCategoriasAR, JuegoMemoriaAR

# ----- ESTADOS DE LA APLICACION -----
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
        self.usuario_nombre = None
        self.usuario_data = None
        self.esperando_comando_sesion = False
        self.gestor_juegos = GestorJuegosAR()
        self.modo_juego = None
        self.idioma_seleccionado = None
        self.idioma_cambiado = None
        self.contador_idioma = 0
        self.nombre_cambiado = None
        self.contador_nombre = 0
        self.vector_facial_actual = None

        # Variables del sistema de progreso
        self.marcadores_detectados = set()  # IDs de marcadores que se han detectado
        self.marcadores_respondidos = set()  # IDs de marcadores que se han respondido correctamente
        self.marcadores_pendientes = set()  # IDs de marcadores que faltan por responder
        self.puntuacion = 0
        self.total_preguntas = 0
        self.juego_completado = False
        self.tiempo_escaneo = 0
        self.en_fase_escaneo = False
        self.error_mensaje = ""
        self.cara_detectada = False
        self.mensaje_temporal = ""

# ----- CONFIGURACIÓN RECONOCIMIENTO FACIAL -----
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ----- VARIABLES GLOBALES -----
JUEGO_ACTUAL = "juego_1"  
state = GameState()
voice_thread_active = False
recognizer = None
microphone = None
# Diccionario para almacenar las escenas de cada modelo
escenas = {}  

# ----- FUNCIONES RELACIONADAS CON EL RECONOCIMIENTO DE VOZ -----
def inicializar_microfono():
    #Inicializa el micrófono de forma no bloqueante
    global recognizer, microphone, state
    
    try:
        print(" Inicializando micrófono...")
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
    """Verificar si la respuesta del usuario es correcta"""
    texto = texto.lower().strip()
    respuesta_correcta = respuesta_correcta.lower().strip()
    
    print(f" Verificando: '{texto}' == '{respuesta_correcta}'")
    
    # Verificación directa
    if respuesta_correcta in texto:
        print(" Coincidencia directa encontrada")
        return True
    
    # Verificación inversa (por si el usuario dice más palabras)
    if texto in respuesta_correcta:
        print(" Coincidencia inversa encontrada")
        return True
    
    # Verificaciones con variaciones y sinónimos
    variaciones = {
        'pera': ['pera', 'peras'],
        'cebolleta': ['cebolleta', 'cebolletas', 'cebollín', 'cebollino'],
        'cebolla': ['cebolla', 'cebollas'],
        'lechuga': ['lechuga', 'lechugas'],
        'limon': ['limón', 'limon', 'limones'],
        'limón': ['limón', 'limon', 'limones'],
        'pimiento rojo': ['pimiento rojo', 'pimiento', 'pimientos rojos', 'pimientos'],
        'pimiento verde': ['pimiento verde', 'pimiento', 'pimientos verdes', 'pimientos'],
        'pimiento': ['pimiento', 'pimientos', 'pimiento rojo', 'pimiento verde'],
        'uvas': ['uvas', 'uva', 'racimo', 'racimo de uvas'],
        'uva': ['uvas', 'uva', 'racimo', 'racimo de uvas'],
        'zanahoria': ['zanahoria', 'zanahorias'],
    }
    
    # Buscar en variaciones para la respuesta correcta
    if respuesta_correcta in variaciones:
        for variacion in variaciones[respuesta_correcta]:
            if variacion in texto:
                print(f" Variación encontrada: '{variacion}' en '{texto}'")
                return True
    
    # Buscar si alguna clave de variaciones está en el texto del usuario
    for clave, lista_variaciones in variaciones.items():
        if clave == respuesta_correcta:
            continue
        for variacion in lista_variaciones:
            if variacion in texto and clave == respuesta_correcta:
                print(f" Variación de clave encontrada: '{variacion}' -> '{clave}'")
                return True
    
    # Verificación de palabras individuales (para casos complejos)
    palabras_texto = texto.split()
    palabras_respuesta = respuesta_correcta.split()
    
    # Si la respuesta correcta es una sola palabra
    if len(palabras_respuesta) == 1:
        palabra_correcta = palabras_respuesta[0]
        for palabra in palabras_texto:
            if palabra == palabra_correcta:
                print(f" Palabra individual encontrada: '{palabra}'")
                return True
            # Verificar en variaciones también
            if palabra_correcta in variaciones:
                if palabra in variaciones[palabra_correcta]:
                    print(f" Variación de palabra encontrada: '{palabra}' -> '{palabra_correcta}'")
                    return True
    
    print(f" No se encontró coincidencia para '{texto}' vs '{respuesta_correcta}'")
    return False

def reconocimiento_voz():
    global state, recognizer, microphone, voice_thread_active
    
    voice_thread_active = True
    
    while voice_thread_active:
        if state.esperando_voz and state.microfono_listo and recognizer and microphone:
            try:
                print(f" Escuchando en fase: {state.fase}")
                
                # Ajustar tiempo según la fase
                timeout = 5 if "nombre" in state.fase else 3
                phrase_limit = 6 if "nombre" in state.fase else 4
                
                with microphone as source:
                    audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
                
                texto = recognizer.recognize_google(audio, language="es-ES").lower().strip()
                print(f" Detectado: '{texto}'")
                
                # Limpiar mensaje de error previo
                if hasattr(state, 'error_mensaje'):
                    delattr(state, 'error_mensaje')
                
                # ===== MANEJO DE COMANDOS INICIALES =====
                if state.fase == "esperando_comando":
                    if "iniciar sesión" in texto:
                        print(" Comando: Iniciar sesión")
                        state.fase = "intentando_iniciar_sesion"

                    elif "registrar" in texto or "registrarme" in texto:
                        print(" Comando: Registro")
                        state.fase = "esperando_nombre_registro"

                    elif "iniciar" in texto:
                        
                        print("❓ ¿Quieres iniciar sesión o registrarte?")
                        state.error_mensaje = "Di 'iniciar sesión' o 'registrarme' para continuar."

                    state.esperando_voz = False

                # ===== INICIO DE SESIÓN =====
                elif state.fase == "intentando_iniciar_sesion":
                    if hasattr(state, 'vector_facial_actual'):
                        nombre_encontrado, datos_usuario = buscar_usuario_por_cara(state.vector_facial_actual)
                        if nombre_encontrado:
                            state.usuario_nombre = nombre_encontrado
                            state.usuario_data = datos_usuario
                            state.sesion_iniciada = True
                            state.fase = "menu_principal"
                            print(f" Sesión iniciada para {nombre_encontrado}")
                        else:
                            print(" Cara no registrada.")
                            state.fase = "inicio_sesion_fallido"
                    else:
                        print(" No se ha detectado una cara para verificar.")
                        state.error_mensaje = "No se ha detectado una cara para iniciar sesion."

                    state.esperando_voz = False

                # ===== REGISTRO DE NOMBRE =====
                elif state.fase == "esperando_nombre_registro":
                    nombre = texto.strip().title()
                    print(f" Usuario dijo llamarse: {nombre}")
                    
                    if len(nombre) >= 2:  # Validación básica
                        # Verificar si el usuario ya existe
                        if verificar_usuario_existe(nombre):
                            state.error_mensaje = f"Usuario {nombre} ya existe. Di otro nombre"
                            print(f" Usuario {nombre} ya existe")
                        else:
                            state.usuario_nombre = nombre
                            state.esperando_voz = False
                            state.fase = "esperando_idioma_registro"
                    else:
                        state.error_mensaje = "Nombre muy corto, intenta de nuevo"
                
                # ===== SELECCIÓN DE IDIOMA EN REGISTRO =====
                elif state.fase == "esperando_idioma_registro":
                    print(f" Usuario eligio idioma: {texto}")
                    
                    idiomas_disponibles = {
                        "español": "es",
                        "castellano": "es", 
                        "espanol": "es",
                        "inglés": "en",
                        "ingles": "en",
                        "english": "en"
                    }
                    
                    idioma_codigo = idiomas_disponibles.get(texto.strip())
                    
                    if idioma_codigo:
                        # Registrar usuario con vector facial
                        if hasattr(state, 'vector_facial_actual'):
                            datos = registrar_usuario(
                                state.usuario_nombre, 
                                idioma_codigo, 
                                state.vector_facial_actual
                            )
                            
                            if datos:
                                print(f" Usuario {state.usuario_nombre} registrado correctamente")
                                state.usuario_data = datos
                                state.esperando_voz = False
                                state.fase = "menu_principal"
                                state.registro_exitoso = True
                                state.idioma_seleccionado = texto.strip()
                                state.esperando_voz = False
                            else:
                                state.error_mensaje = "Error registrando usuario"
                        else:
                            state.error_mensaje = "Error: no hay datos faciales"
                    else:
                        print(f" Idioma no reconocido: {texto}")
                        state.error_mensaje = "Idioma no valido (di: espaniol o ingles)"

                # ===== MENÚ PRINCIPAL =====
                elif state.fase == "menu_principal":
                    if "comenzar" in texto or "empezar" in texto or "jugar" in texto:
                        print("Iniciando selección de modo de juego")
                        state.esperando_voz = False
                        state.fase = "seleccion_modo"
                    elif "cuenta" in texto or "personal" in texto:
                        print("Entrando en la configuracion de la cuenta...")
                        state.esperando_voz = False
                        state.fase = "configuracion_cuenta"
                    elif "progreso" in texto or "estadisticas" in texto:
                        print("Entrando al progreso...")
                        state.esperando_voz = False
                        state.fase = "ver_progreso"
                    elif "salir" in texto or "cerrar" in texto:
                        print(" Cerrando aplicación")
                        state.esperando_voz = False
                        state.fase = "salir"
                
                # ===== PROGRESO =====
                elif state.fase == "ver_progreso":
                    if "volver" in texto or "atras" in texto or "salir" in texto:
                        print("Volviendo al menu principal")
                        state.esperando_voz = False
                        state.fase = "menu_principal"

                # ===== CONFIGURACIÓN CUENTA =====
                elif state.fase == "configuracion_cuenta":

                    if texto.strip().lower() == "cambiar nombre":
                        state.fase = "esperando_nuevo_nombre"
                        state.esperando_voz = True
                        state.mensaje_temporal = "Di tu nuevo nombre"

                    elif texto.strip().lower() == "cambiar idioma":
                        state.fase = "esperando_nuevo_idioma"
                        state.esperando_voz = True
                        state.mensaje_temporal = "Di el nuevo idioma (espaniol o ingles)"

                    elif texto.strip().lower() == "volver":
                        state.fase = "menu_principal"
                        state.esperando_voz = True
                
                elif state.fase == "esperando_nuevo_nombre":
                    nuevo_nombre = texto.strip().title()
                    print(f" Usuario quiere cambiar su nombre a: {nuevo_nombre}")

                    if len(nuevo_nombre) >= 2:
                        comandos_reservados = ["salir", "cuenta", "comenzar", "cambiar nombre", "cambiar idioma", "volver"]
                        
                        if verificar_usuario_existe(nuevo_nombre) and nuevo_nombre != state.usuario_nombre:
                            state.error_mensaje = f"El nombre {nuevo_nombre} ya esta en uso"
                        elif nuevo_nombre.lower() in comandos_reservados:
                            state.error_mensaje = f"'{nuevo_nombre}' no es un nombre valido"
                        else:
                            nombre_anterior = state.usuario_nombre
                            state.usuario_nombre = nuevo_nombre
                            state.fase = "configuracion_cuenta"
                            state.nombre_cambiado = True
                            state.contador_nombre = 0
                            state.esperando_voz = False
                            actualizar_nombre_usuario(nombre_anterior, nuevo_nombre)
                            print(f" Nombre cambiado a {nuevo_nombre}")
                        
                        state.nombre_cambiado = False
                    else:
                        state.error_mensaje = "Nombre muy corto, intenta de nuevo"
                
                elif state.fase == "esperando_nuevo_idioma":
                    idiomas_disponibles = {
                        "español": "es",
                        "castellano": "es",
                        "espanol": "es",
                        "inglés": "en",
                        "english": "en"
                    }

                    idioma_texto = texto.strip().lower()
                    idioma_codigo = idiomas_disponibles.get(idioma_texto)

                    if idioma_codigo:
                        state.usuario_data['idioma'] = idioma_codigo
                        state.idioma_seleccionado = idioma_texto
                        state.fase = "configuracion_cuenta"
                        state.idioma_cambiado = True
                        state.contador_idioma = 0
                        state.esperando_voz = False
                        if state.idioma_seleccionado in ["español", "castellano", "espanol"]:
                            state.idioma_seleccionado = "es"
                        else:
                            state.idioma_seleccionado = "en"

                        actualizar_idioma_usuario(state.usuario_nombre, state.idioma_seleccionado)
                        print(f" Idioma cambiado a {idioma_texto}")
                        state.idioma_cambiado = False
                    else:
                        state.error_mensaje = "Idioma no valido (di: espaniol o ingles)"

                # ===== SELECCIÓN DE MODO =====
                elif state.fase == "seleccion_modo":
                    if "entrenamiento" in texto or "entrenar" in texto or "practicar" in texto:
                        print(" Modo entrenamiento seleccionado")
                        state.esperando_voz = False
                        state.fase = "seleccion_juego"
                        state.modo_juego = "entrenamiento"
                    elif "evaluación" in texto or "evaluacion" in texto or "evaluar" in texto:
                        print(" Modo evaluación seleccionado")
                        state.esperando_voz = False
                        state.fase = "seleccion_juego"
                        state.modo_juego = "evaluacion"
                    elif "volver" in texto or "atras" in texto or "salir" in texto:
                        print("Volviendo al menu principal")
                        state.esperando_voz = False
                        state.fase = "menu_principal"
                
                # ===== SELECCIÓN DE JUEGO =====
                elif state.fase == "seleccion_juego":
                    juego_iniciado = False
                    
                    if state.modo_juego == "entrenamiento":
                        if "descubre" in texto or "nombra" in texto or "nombres" in texto:
                            print(" Juego 'Descubre y Nombra' seleccionado")
                            try:
                                if not hasattr(state, 'gestor_juegos') or state.gestor_juegos is None:
                                    print(" Error: gestor_juegos no está inicializado")
                                elif "volver" in texto or "atras" in texto or "salir" in texto:
                                    print("Volviendo al menu de juego")
                                    state.esperando_voz = False
                                    state.fase = "seleccion_juego"
                                else:
                                    state.gestor_juegos.establecer_modo(state.modo_juego)
                                    print(f" Modo establecido en gestor: {state.modo_juego}")
                                    
                                    state.gestor_juegos.iniciar_juego("descubre")
                                    
                                    if (state.gestor_juegos.juego_activo is not None and 
                                        state.gestor_juegos.estado_juego == "en_juego"):
                                        juego_iniciado = True
                                        print(" Juego iniciado correctamente")
                                    else:
                                        print(" Error: juego no se inició correctamente")
                                    
                            except Exception as e:
                                print(f" Error detallado al iniciar juego: {e}")
                                import traceback
                                print(f"   - Traceback: {traceback.format_exc()}")
                        
                        elif "frutas" in texto or "encuentra" in texto:
                            print("Juego 'Encuentra las Frutas' seleccionado")
                            try:
                                if not hasattr(state, 'gestor_juegos') or state.gestor_juegos is None:
                                    print("Error: gestor_juegos no esta inicializado")
                                elif "volver" in texto or "atras" in texto or "salir" in texto:
                                    print("Volviendo al menu de juego")
                                    state.esperando_voz = False
                                    state.fase = "seleccion_juego"
                                else:
                                    state.gestor_juegos.establecer_modo(state.modo_juego)
                                    state.gestor_juegos.iniciar_juego("frutas")
                                    
                                    if (state.gestor_juegos.juego_activo is not None and 
                                        state.gestor_juegos.estado_juego == "en_juego"):
                                        juego_iniciado = True
                                        print(" Juego iniciado correctamente")
                                    else:
                                        print(" Error: juego no se inicio correctamente")
                              
                            except Exception as e:
                                print(f" Error detallado al iniciar juego: {e}")
                                import traceback
                                print(f"   - Traceback: {traceback.format_exc()}")
                        
                        elif "volver" in texto or "atras" in texto or "salir" in texto:
                            print("Volviendo a menu de modos de juegos")
                            state.esperando_voz = False
                            state.fase = "seleccion_modo"

                    elif state.modo_juego == "evaluacion":
                        if "categorias" in texto or "categorías" in texto or "agrupa" in texto or "separa" in texto:
                            print(" Juego 'Agrupa por Categorías' seleccionado")
                            try:
                                if not hasattr(state, 'gestor_juegos') or state.gestor_juegos is None:
                                    print(" Error: gestor_juegos no esta inicializado")
                                elif "volver" in texto or "atras" in texto or "salir" in texto:
                                    print("Volviendo al menu de juego")
                                    state.esperando_voz = False
                                    state.fase = "seleccion_juego"
                                else:
                                    state.gestor_juegos.establecer_modo(state.modo_juego)
                                    state.gestor_juegos.iniciar_juego("categorias")
                                    
                                    if (state.gestor_juegos.juego_activo is not None and 
                                        state.gestor_juegos.estado_juego == "en_juego"):
                                        juego_iniciado = True
                                        print(" Juego iniciado correctamente")
                                    else:
                                        print(" Error: juego no se inicio correctamente")
                                        
                            except Exception as e:
                                print(f" Error detallado al iniciar juego: {e}")
                                import traceback
                                print(f"   - Traceback: {traceback.format_exc()}")
                        
                        elif "memoria" in texto or "recuerda" in texto or "secuencia" in texto:
                            print(" Juego 'Juego de Memoria' seleccionado")
                            try:
                                if not hasattr(state, 'gestor_juegos') or state.gestor_juegos is None:
                                    print(" Error: gestor_juegos no esta inicializado")
                                elif "volver" in texto or "atras" in texto or "salir" in texto:
                                    print("Volviendo al menu de juego")
                                    state.esperando_voz = False
                                    state.fase = "seleccion_juego"
                                else:
                                    state.gestor_juegos.establecer_modo(state.modo_juego)
                                    state.gestor_juegos.iniciar_juego("memoria")
                                    
                                    if (state.gestor_juegos.juego_activo is not None and 
                                        state.gestor_juegos.estado_juego == "en_juego"):
                                        juego_iniciado = True
                                        print(" Juego iniciado correctamente")
                                
                                    else:
                                        print(" Error: juego no se inicio correctamente")
                                        
                            except Exception as e:
                                print(f" Error detallado al iniciar juego: {e}")
                                import traceback
                                print(f"   - Traceback: {traceback.format_exc()}")
                        elif "volver" in texto or "atras" in texto or "salir" in texto:
                            print("Volviendo a menu de modos de juegos")
                            state.esperando_voz = False
                            state.fase = "seleccion_modo"

                    if juego_iniciado:
                        state.esperando_voz = False
                        state.fase = "jugando"
                        print(f" Cambiando a fase 'jugando'")
                    else:
                        print("  No se pudo iniciar el juego. Permaneciendo en seleccion de juego.")
                        state.esperando_voz = True
                
                # ===== COMANDOS DURANTE EL JUEGO =====
                elif state.fase == "jugando":
                    if "salir" in texto or "volver" in texto or "atras" in texto:
                        print(" Volviendo al menu")
                        state.esperando_voz = False
                        state.fase = "menu_principal"
                        if hasattr(state.gestor_juegos, 'resetear'):
                            state.gestor_juegos.resetear()
                    
                    else:
                        # Usar método interno del juego
                        if (hasattr(state.gestor_juegos, 'juego_activo') and
                            hasattr(state.gestor_juegos.juego_activo, 'procesar_comando')):
                            
                            resultado = state.gestor_juegos.juego_activo.procesar_comando(texto)
                            
                            if resultado:
                                print(" Comando de voz procesado por el juego")
                                state.gestor_juegos.procesar_resultado_juego(resultado)
                                state.esperando_voz = False
                            else:
                                print(" Comando no reconocido por el juego")

                # ===== RESPUESTAS DEL JUEGO =====
                elif state.fase == "esperando_respuesta" and hasattr(state, 'info_modelo_actual') and state.info_modelo_actual:
                    respuesta_correcta = state.info_modelo_actual['respuesta_correcta']
                    
                    if verificar_respuesta(texto, respuesta_correcta):
                        state.respuesta_recibida = texto
                        state.respuesta_correcta = True
                        state.puntuacion += 1
                        
                        if hasattr(state, 'marcadores_respondidos') and hasattr(state, 'marker_id_actual'):
                            state.marcadores_respondidos.add(state.marker_id_actual)
                        
                        if hasattr(state, 'marcadores_pendientes') and hasattr(state, 'marker_id_actual'):
                            state.marcadores_pendientes.discard(state.marker_id_actual)
                        
                        print(f" ¡Respuesta correcta! Puntuacion: {state.puntuacion}")
                    else:
                        state.respuesta_recibida = texto
                        state.respuesta_correcta = False
                        print(f" Respuesta incorrecta. Esperaba: {respuesta_correcta}")
                    
                    if hasattr(state, 'total_preguntas'):
                        state.total_preguntas += 1
                    
                    if "volver" in texto or "atras" in texto or "salir" in texto:
                        print("Volviendo al menu principal")
                        state.esperando_voz = False
                        state.fase = "menu_principal"

                    state.fase = "resultado"
                    state.esperando_voz = False
                    state.mostrar_resultado = True
                    state.tiempo_resultado = time.time()
                
                # ===== COMANDOS EN RESULTADOS =====
                elif state.fase == "resultado":
                    if "continuar" in texto or "siguiente" in texto:
                        print("➡️ Continuando al siguiente...")
                        state.esperando_voz = False
                        state.fase = "jugando"
                        state.mostrar_resultado = False
                    if "volver" in texto or "atras" in texto or "salir" in texto:
                        print("Volviendo al menu principal")
                        state.esperando_voz = False
                        state.fase = "menu_principal"
            
            # ===== MANEJO DE ERRORES =====
            except sr.WaitTimeoutError:
                print(" Timeout - reintentando...")
                if hasattr(state, 'fase') and state.fase in ["jugando", "esperando_respuesta"]:
                    state.error_mensaje = "No se escucho respuesta, intenta de nuevo"
                continue
            
            except sr.UnknownValueError:
                print(" No se entendió el audio")
                state.error_mensaje = "No se entendio, habla mas claro"
                continue
            
            except sr.RequestError as e:
                print(f" Error del servicio de reconocimiento: {e}")
                state.error_mensaje = "Error de conexion"
                time.sleep(1)
                continue
            
            except Exception as e:
                print(f" Error inesperado en reconocimiento: {e}")
                import traceback
                print(f"   - Traceback: {traceback.format_exc()}")
                state.error_mensaje = "Error inesperado"
                time.sleep(1)
                continue
        else:
            time.sleep(0.1)
 
# ----- FUNCION PARA INCIAR JUEGOS -----
def iniciar_juego(juego_id):
    global state, JUEGO_ACTUAL
    JUEGO_ACTUAL = juego_id
    state.fase = "escaneo_inicial"
    state.tiempo_escaneo = time.time()

    # Cargar progreso si existe
    juego_data = state.usuario_data.get("juegos", {}).get(juego_id, {})
    state.marcadores_respondidos = set(juego_data.get("respondidos", []))
    state.puntuacion = juego_data.get("puntuacion", 0)
    print(f" Juego {juego_id} iniciado con progreso cargado")

# ----- FUNCION DE PROCESAMIENTO FACIAL -----
def extraer_vector_facial(frame, face_box):
    """Extrae el vector de características faciales usando face_recognition"""
    try:
        x, y, w, h = face_box
        # Convertir de BGR a RGB para face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Obtener encodings faciales
        face_locations = [(y, x+w, y+h, x)]  # formato: (top, right, bottom, left)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) > 0:
            return face_encodings[0].tolist()  # Convertir a lista para JSON
        return None
    except Exception as e:
        print(f"Error extrayendo vector facial: {e}")
        return None
           
# ----- FUNCION PARA DETECTAR MARCADORES DISPONIBLES -----
def detectar_marcadores_disponibles(frame, detector, cameraMatrix, distCoeffs):
    ret, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)
    marcadores_encontrados = set()
    
    if ret and pose is not None:
        for marker_id in pose.keys():
            if marker_id in MODELOS_FRUTAS_VERDURAS:
                marcadores_encontrados.add(marker_id)
    
    return marcadores_encontrados

# ----- FUNCION DE REALIDAD AUMENTADA -----
def realidad_mixta(frame, detector, cameraMatrix, distCoeffs):
    global state, escenas

    # Detectar marcadores disponibles en cada frame
    marcadores_actuales = detectar_marcadores_disponibles(frame, detector, cameraMatrix, distCoeffs)

    # Solo actualizar marcadores_detectados durante el escaneo inicial
    if state.fase == "escaneo_inicial":
        state.marcadores_detectados.update(marcadores_actuales)

    # Solo mostrar modelos si el usuario está identificado y estamos en una fase activa
    if state.usuario_identificado and state.fase in ["pregunta", "esperando_respuesta", "resultado"]:
        ret, pose = detectar_pose(frame, 0.19, detector, cameraMatrix, distCoeffs)

        if ret and pose is not None:
            marker_ids = list(pose.keys())

            if marker_ids:
                # Priorizar marcador pendiente visible
                marcador_pendiente_visible = None
                for mid in marker_ids:
                    if mid in state.marcadores_pendientes:
                        marcador_pendiente_visible = mid
                        break

                # Seleccionar el marcador adecuado
                if marcador_pendiente_visible:
                    marker_id = marcador_pendiente_visible
                else:
                    marker_id = marker_ids[0]

                # Solo actualizar si cambia
                if state.marker_id_actual != marker_id:
                    state.marker_id_actual = marker_id
                    state.info_modelo_actual = obtener_info_modelo(marker_id)

                    # Crear escena si no existe
                    if marker_id not in escenas:
                        modelo = crear_modelo_por_id(marker_id)
                        escenas[marker_id] = crear_escena(modelo, cameraMatrix,
                                                          int(frame.shape[1]), int(frame.shape[0]))

                    print(f" Mostrando: {state.info_modelo_actual['nombre']} (ID: {marker_id})")
                
                

                # Renderizar modelo actual si está disponible
                if state.marker_id_actual in escenas:
                    M = from_opencv_to_pygfx(pose[state.marker_id_actual][0], pose[state.marker_id_actual][1])
                    escenas[state.marker_id_actual].actualizar_camara(M)
                    imagen_render = escenas[state.marker_id_actual].render()
                    imagen_render_bgr = cv2.cvtColor(imagen_render, cv2.COLOR_RGBA2BGRA)
                    #resultado = cuia.alphaBlending(imagen_render_bgr, frame)
                    frame_con_modelo = cuia.alphaBlending(imagen_render_bgr, frame.copy())
                    return frame_con_modelo

    return frame

# ----- FUNCION PARA DIBUJAR TEXTO EN EL FRAME -----
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

# ----- FUNCION PRINCIPAL -----
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
    #ar.process = lambda frame: realidad_mixta(frame, detector, cameraMatrix, distCoeffs)
    ar.process = lambda frame: realidad_mixta(frame.copy(), detector, cameraMatrix, distCoeffs)
 
    # Inicializar micrófono en hilo separado
    hilo_microfono = threading.Thread(target=inicializar_microfono, daemon=True)
    hilo_microfono.start()

    # Iniciar hilo de reconocimiento de voz
    hilo_voz = threading.Thread(target=reconocimiento_voz, daemon=True)
    hilo_voz.start()

    print("🎮 Kids&Veggies iniciado - Mira a la camara para comenzar")
    print(" Marcadores disponibles:")
    for marker_id, info in MODELOS_FRUTAS_VERDURAS.items():
        print(f"   ID {marker_id}: {info['nombre']} ({info['tipo']})")

    try:
        while True:
            ret, frame = ar.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_time = time.time()

            alto = frame.shape[0]  # Altura del frame
            
            # ----- FASE 1: Reconocimiento Facial inicial -----
            if state.fase == "reconocimiento_facial":
                faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Detectar rostro más grande
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    (x, y, w, h) = faces[0]
                    
                    # Dibujar rectángulo
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Extraer vector facial
                    vector_facial = extraer_vector_facial(frame, (x, y, w, h))
                    
                    if vector_facial is not None:
                        # Guardar vector actual para uso posterior
                        state.vector_facial_actual = vector_facial
                        state.cara_detectada = True
                        
                        draw_text_with_background(frame, "¡Cara detectada!", (x, y-15), 
                                                color=(0, 255, 0), bg_color=(0, 100, 0))
                        draw_text_with_background(frame, "Di: 'iniciar sesion' o 'registrarme'", (50, 50),
                                                color=(255, 255, 255), bg_color=(0, 0, 100))
                        
                        if state.fase != "esperando_comando":
                            state.fase = "esperando_comando"
                            state.esperando_voz = True
                    else:
                        draw_text_with_background(frame, "Error procesando cara", (x, y-15), 
                                                color=(255, 255, 255), bg_color=(200, 50, 50))
                else:
                    draw_text_with_background(frame, "Mira a la camara para comenzar", (50, 50),
                                            color=(255, 255, 255), bg_color=(100, 100, 100))
                    state.cara_detectada = False
                
                # Mostrar estado del micrófono
                if state.microfono_listo:
                    draw_text_with_background(frame, " Microfono listo ", (50, 100),
                                            color=(0, 255, 0), bg_color=(0, 100, 0))
                else:
                    draw_text_with_background(frame, " Configurando microfono... ", (50, 100),
                                            color=(255, 255, 0), bg_color=(100, 100, 0))

            # ----- FASE 2: Esperar comando de voz -----
            elif state.fase == "esperando_comando":
                draw_text_with_background(frame, "BIENVENIDO A LA PLATAFORMA EDUCATIVA DE Kids&Veggies", (50, 60),
                    color=(255, 255, 255), bg_color=(56,118, 29))
                draw_text_with_background(frame, "Di: 'iniciar sesion' o 'registrarme'", (50, 100),
                    color=(255, 255, 255), bg_color=(0, 0, 100))
                draw_text_with_background(frame, "Esperando comando de voz...", (50, 140),
                    color=(255, 255, 0), bg_color=(100, 100, 0))
                
                if not state.esperando_voz:
                    state.esperando_voz = True

            # ----- FASE 3: Proceso de inicio de sesión -----
            elif state.fase == "intentando_iniciar_sesion":
                draw_text_with_background(frame, "Verificando identidad...", (50, alto - 100),
                                        color=(255, 255, 255), bg_color=(0, 100, 100))
                
                # Buscar usuario por cara
                if hasattr(state, 'vector_facial_actual'):
                    nombre_encontrado, datos_usuario = buscar_usuario_por_cara(state.vector_facial_actual)
                    
                    if nombre_encontrado:
                        # Usuario encontrado - iniciar sesión
                        state.usuario_nombre = nombre_encontrado
                        state.usuario_data = datos_usuario
                        state.fase = "menu_principal"
                        state.sesion_iniciada = True
                        print(f"🔓 Sesión iniciada para {nombre_encontrado}")
                    else:
                        print("❌ Cara no registrada. Debes registrarte primero.")
                        state.error_mensaje = "Cara no registrada. Di 'registrarme' para crear una cuenta."
                        state.fase = "inicio_sesion_fallido"
                        state.esperando_voz = False

            # ----- FASE 4: Inicio de sesión fallido -----
            elif state.fase == "inicio_sesion_fallido":
                draw_text_with_background(frame, "Cara no registrada. No puedes iniciar sesion.", (50, alto - 130),
                                        color=(255, 255, 255), bg_color=(200, 50, 50))
                draw_text_with_background(frame, "Di: 'registrarme' para crear una cuenta", (50, alto - 100),
                                        color=(255, 255, 255), bg_color=(0, 0, 100))
                
                # Guardar el tiempo de inicio de esta fase si no existe
                if not hasattr(state, 'tiempo_pausa_cara_no_registrada'):
                    state.tiempo_pausa_cara_no_registrada = time.time()
                
                # Comprobar si pasaron 2 segundos para avanzar
                if time.time() - state.tiempo_pausa_cara_no_registrada > 2:
                    state.fase ="esperando_comando"
                    del state.tiempo_pausa_cara_no_registrada
                
                state.esperando_voz = True

            # ----- FASE 5: Proceso de registro fallido o no -----
            elif state.fase == "registro_denegado_por_seguridad":
                draw_text_with_background(frame, "Esa cara ya esta registrada.", (50, alto - 130),
                                        color=(255, 255, 255), bg_color=(200, 50, 50))
                draw_text_with_background(frame, "Di: 'iniciar sesion' para entrar", (50, alto - 100),
                                        color=(255, 255, 255), bg_color=(0, 0, 100))
                # Guardar el tiempo de inicio de esta fase si no existe
                if not hasattr(state, 'tiempo_pausa'):
                    state.tiempo_pausa = time.time()
                
                # Comprobar si pasaron 2 segundos para avanzar
                if time.time() - state.tiempo_pausa > 2:
                    state.fase = "esperando_comando"
                    del state.tiempo_pausa
                    
                state.esperando_voz = True
            
            # ----- Proceso de registro -----
            elif state.fase == "esperando_nombre_registro":
                # Verificar si la cara ya está registrada
                if hasattr(state, 'vector_facial_actual'):
                    nombre_encontrado, datos_usuario = buscar_usuario_por_cara(state.vector_facial_actual)
                    if nombre_encontrado:
                        # Ya existe, no dejar registrar
                        state.error_mensaje = f"Usuario registrado como {nombre_encontrado}. Di 'iniciar sesion'."
                        print(f" Intento de registro con cara ya registrada: {nombre_encontrado}")

                        # Guardar el tiempo de inicio de esta fase si no existe
                        if not hasattr(state, 'tiempo_pausa'):
                            state.tiempo_pausa = time.time()
                        
                        # Comprobar si pasaron 2 segundos para avanzar
                        if time.time() - state.tiempo_pausa > 2:
                            state.fase = "registro_denegado_por_seguridad"
                            del state.tiempo_pausa
                        
                        state.esperando_voz = False
                    else:
                        # Continuar con el flujo de pedir nombre
                        draw_text_with_background(frame, "Dime tu nombre...", (50, alto - 100),
                                                color=(255, 255, 255), bg_color=(0, 100, 0))
                        
                        if hasattr(state, 'error_mensaje'):
                            draw_text_with_background(frame, state.error_mensaje, (50, alto - 130),
                                                    color=(255, 255, 255), bg_color=(200, 50, 50))
                        state.esperando_voz = True
                else:
                    state.error_mensaje = "No se detecto tu cara. Intenta de nuevo."
                    print(" Registro detenido: no hay vector facial.")

            elif state.fase == "esperando_idioma_registro":
                draw_text_with_background(frame, f"Hola {state.usuario_nombre}!", (50, alto - 160),
                                color=(255, 255, 255), bg_color=(0, 100, 0))
                draw_text_with_background(frame, "¿Que idioma prefieres?", (50, alto - 130),
                                color=(255, 255, 255), bg_color=(100, 100, 0))
                draw_text_with_background(frame, "Di: 'espaniol' o 'ingles'", (50, alto - 100),
                                color=(255, 255, 255), bg_color=(100, 100, 0))
                
                state.esperando_voz = True
                
                if hasattr(state, 'error_mensaje'):
                    draw_text_with_background(frame, state.error_mensaje, (50, alto - 190),
                                            color=(255, 255, 255), bg_color=(200, 50, 50))
            
            # ----- FASE 6: Menu principal -----
            elif state.fase == "menu_principal":
                # Mostrar mensaje de registro exitoso si acabamos de registrar
                if hasattr(state, 'registro_exitoso') and state.registro_exitoso:
                    draw_text_with_background(frame, f"Usuario registrado exitosamente", (50, alto - 220),
                                    color=(255, 255, 255), bg_color=(50, 200, 50))
                    if hasattr(state, 'idioma_seleccionado'):
                        draw_text_with_background(frame, f"Idioma: {state.idioma_seleccionado}", (50, alto - 250),
                                        color=(255, 255, 255), bg_color=(50, 200, 50))
                    
                    # Contador simple para limpiar el mensaje
                    if not hasattr(state, 'contador_registro'):
                        state.contador_registro = 0
                    state.contador_registro += 1
                    
                    if state.contador_registro > 90:  # ~3 segundos a 30fps
                        delattr(state, 'registro_exitoso')
                        delattr(state, 'contador_registro')
                        if hasattr(state, 'idioma_seleccionado'):
                            delattr(state, 'idioma_seleccionado')
                
                # Menú principal
                draw_text_with_background(frame, "MENU PRINCIPAL", (50, 40),
                                color=(255, 255, 255), bg_color=(0, 100, 0))
                
                nombre = state.usuario_nombre or "Usuario"
                draw_text_with_background(frame, f"Bienvenido, {nombre}", (50, 80),
                                font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 100))
                
                draw_text_with_background(frame, "Di 'cuenta' para administrar tus datos", (50, 120),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "Di 'progreso' para ver tus estadisticas", (50, 160),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "Di 'comenzar' para jugar", (50, 200),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "Di 'salir' para cerrar", (50, 240),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
                
                # Activar reconocimiento de voz
                state.esperando_voz = True

            # ----- FASE 6.1: Acceder a ver progreso -----
            elif state.fase == "ver_progreso":
                
                progreso = obtener_progreso_usuario(state.usuario_nombre)
                
                if not progreso or "mensaje" in progreso:
                    draw_text_with_background(frame, "No hay datos de progreso aun.", (50, 100),
                                            color=(255, 255, 255), bg_color=(100, 0, 0))
                else:
                    y = 40
                    draw_text_with_background(frame, f"PROGRESO DE {progreso['nombre'].upper()}", (50, y), bg_color=(0, 100, 100)); y += 40
                    draw_text_with_background(frame, f"Partidas totales: {progreso['total_partidas']}", (50, y)); y += 30
                    draw_text_with_background(frame, f"Numeros de juegos diferentes: {progreso['juegos_completados']}", (50, y)); y += 30

                    for modo, stats in progreso["resumen_por_modo"].items():
                        draw_text_with_background(frame, f"--- MODO {modo.upper()} ---", (50, y), bg_color=(20, 20, 100)); y += 30
                        draw_text_with_background(frame, f"  Juegos: {stats['juegos_jugados']}", (50, y)); y += 25
                        draw_text_with_background(frame, f"  Partidas: {stats['total_partidas_modo']}", (50, y)); y += 25
                        draw_text_with_background(frame, f"  Promedio: {stats['promedio_general']:.1f}%", (50, y)); y += 25

                        for juego, info in stats["juegos"].items():
                            draw_text_with_background(frame, f"{juego}: {info['puntuacion_media']:.1f}% ({info['partidas_jugadas']} partidas)", (50, y)); y += 30

                draw_text_with_background(frame, "Di 'volver' para regresar al menu", (50, alto - 60), bg_color=(0, 0, 100))
                state.esperando_voz = True

            # ----- FASE 6.2: Acceder a la configuración de la cuenta -----
            elif state.fase == "configuracion_cuenta":
                draw_text_with_background(frame, "CONFIGURACION DE LA CUENTA", (50, 40),
                                color=(255, 255, 255), bg_color=(0, 100, 0))
                
                nombre = state.usuario_nombre or "Usuario"
                idioma = state.usuario_data.get("idioma")
                if idioma == "es":
                    idioma = "Español"
                else: 
                    idioma = "Inglés"
                
                draw_text_with_background(frame, f"Nombre: {nombre}", (50, 80),
                                font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 100))
                draw_text_with_background(frame, f"Idioma: {idioma}", (50, 120),
                                font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 120))
                
                draw_text_with_background(frame, "Di 'cambiar nombre' para modificar tu nombre", (50, 160),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "Di 'cambiar idioma' para seleccionar un nuevo idioma", (50, 200),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
                draw_text_with_background(frame, "Di 'volver' para regresar al menu principal", (50, 240),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
               
                if hasattr(state, 'error_mensaje'):
                    draw_text_with_background(frame, state.error_mensaje, (50, alto - 190),
                                            color=(255, 255, 255), bg_color=(200, 50, 50))

                # Activar reconocimiento de voz
                state.esperando_voz = True

             # ----- FASE 6.2: Acceder a la configuración de la cuenta -----
            
            # ----- FASE 6.3: Cambio de nombre -----
            elif state.fase == "esperando_nuevo_nombre":
                draw_text_with_background(frame, "Escuchando nuevo nombre...", (50, 40),
                                color=(255, 255, 255), bg_color=(0, 100, 0))
                
                # Mostrar mensajes de éxito si se cambio nombre
                if hasattr(state, 'nombre_cambiado') and state.nombre_cambiado:
                    draw_text_with_background(frame, "Nombre actualizado correctamente", (50, alto - 220),
                        color=(255, 255, 255), bg_color=(50, 200, 50))
                    state.contador_nombre += 1
                    if state.contador_nombre > 90:
                        delattr(state, 'nombre_cambiado')
                        delattr(state, 'contador_nombre')
                
                # Mostrar guía si el sistema está esperando nombre/idioma nuevo
                if hasattr(state, 'mensaje_temporal'):
                    draw_text_with_background(frame, state.mensaje_temporal, (50, alto - 280),
                        color=(255, 255, 255), bg_color=(100, 100, 0))
                    delattr(state, 'mensaje_temporal')
                
                if hasattr(state, 'error_mensaje'):
                    draw_text_with_background(frame, state.error_mensaje, (50, alto - 190),
                                            color=(255, 255, 255), bg_color=(200, 50, 50))

            # ----- FASE 6.4: Cambio de idioma -----
            elif state.fase == "esperando_nuevo_idioma":
                draw_text_with_background(frame, "Escuchando nuevo idioma...", (50, 40),
                                color=(255, 255, 255), bg_color=(0, 100, 0))
                # Mostrar mensajes de éxito si se cambio idioma
                if hasattr(state, 'idioma_cambiado') and state.idioma_cambiado:
                    draw_text_with_background(frame, "Idioma actualizado correctamente", (50, alto - 250),
                        color=(255, 255, 255), bg_color=(50, 200, 50))
                    state.contador_idioma += 1
                    if state.contador_idioma > 90:
                        delattr(state, 'idioma_cambiado')
                        delattr(state, 'contador_idioma')
                
                # Mostrar guía si el sistema está esperando nombre/idioma nuevo
                if hasattr(state, 'mensaje_temporal'):
                    draw_text_with_background(frame, state.mensaje_temporal, (50, alto - 280),
                        color=(255, 255, 255), bg_color=(100, 100, 0))
                    delattr(state, 'mensaje_temporal')
                
                if hasattr(state, 'error_mensaje'):
                    draw_text_with_background(frame, state.error_mensaje, (50, alto - 190),
                                            color=(255, 255, 255), bg_color=(200, 50, 50))
                    
            
            # ----- FASE 7: Menú para seleccionar modo de juego -----
            elif state.fase == "seleccion_modo":
                nombre = state.usuario_nombre or "Desconocido"
                draw_text_with_background(frame, f"Hola, {nombre}", (50, 80),
                        font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0))

                # Opciones de modo
                draw_text_with_background(frame, "SELECCIONA MODO DE JUEGO:", (50, 120),
                                        color=(255, 255, 255), bg_color=(0, 100, 0))
                draw_text_with_background(frame, "Di 'entrenamiento' para practicar", (50, 160),
                                        color=(255, 255, 255), bg_color=(0, 0, 100))
                draw_text_with_background(frame, "Di 'evaluacion' para ser evaluado", (50, 200),
                                        color=(255, 255, 255), bg_color=(100, 0, 0))
                draw_text_with_background(frame, "Di 'volver' para regresar al menu principal", (50, 240),
                                color=(255, 255, 255), bg_color=(100, 0, 100))
                
                # Activar reconocimiento de voz
                state.esperando_voz = True

            # ----- FASE 8: Menú para seleccionar el juego -----
            elif state.fase == "seleccion_juego":
                # Mostrar el modo seleccionado
                modo = getattr(state, 'modo_juego', 'desconocido')
                draw_text_with_background(frame, f"MODO: {modo.upper()}", (50, 50),
                                        color=(255, 255, 255), bg_color=(100, 0, 100))
                state.gestor_juegos.establecer_modo(modo)
                
                # Usar el gestor de juegos para mostrar juegos disponibles
                juegos_disponibles = state.gestor_juegos.obtener_juegos_disponibles()
                
                draw_text_with_background(frame, "SELECCIONA UN JUEGO:", (50, 100),
                                        color=(255, 255, 255), bg_color=(0, 100, 0))
                
                y_pos = 140
                for key, juego in juegos_disponibles.items():
                    draw_text_with_background(frame, f"Di '{juego['comando']}' - {juego['nombre']}", 
                                            (50, y_pos), font_scale=0.7,
                                            color=(255, 255, 255), bg_color=(0, 0, 100))
                    y_pos += 30
                    draw_text_with_background(frame, f"  {juego['descripcion']}", 
                                            (70, y_pos), font_scale=0.5,
                                            color=(200, 200, 200), bg_color=(50, 50, 50))
                    y_pos += 40
                
                draw_text_with_background(frame, "Di 'volver' para cambiar modo", (50, y_pos),
                                        font_scale=0.6, color=(255, 255, 0), bg_color=(100, 100, 0))
                
                # Activar reconocimiento de voz
                state.esperando_voz = True
            
            # ----- FASE 9: Jugando al juego -----
            elif state.fase == "jugando":
                # --- 1. Crear dos frames: uno para deteccion, otro para visualizacion ---
                frame_limpio = frame.copy()   # Sin modificar, para deteccion y pose
                frame_visual = frame.copy()   # Aqui ocultaremos marcadores visualmente para mostrar al usuario

                # --- 2. Detectar marcadores sobre frame limpio ---
                marcadores_actuales = detectar_marcadores_disponibles(frame_limpio, detector, cameraMatrix, distCoeffs)
                state.marcadores_detectados.update(marcadores_actuales)

                # --- 3. Ocultar visualmente los marcadores solo en el frame_visual ---
                ocultar_marcadores_visualmente(frame_visual, detector)

                # --- 4. Renderizar modelos 3D usando poses calculadas en frame_limpio ---
                juego_actual = getattr(state.gestor_juegos, 'juego_activo', None)

                if juego_actual and (isinstance(juego_actual, JuegoDescubreAR) or 
                                    isinstance(juego_actual, JuegoMemoriaAR) or 
                                    isinstance(juego_actual, JuegoEncuentraFrutasAR) or 
                                    isinstance(juego_actual, JuegoCategoriasAR)):
                    
                    marcadores_a_renderizar = []
                    if hasattr(juego_actual, 'obtener_marcadores_renderizado'):
                        marcadores_a_renderizar = juego_actual.obtener_marcadores_renderizado()

                    ret_pose, pose = detectar_pose(frame_limpio, 0.19, detector, cameraMatrix, distCoeffs)

                    if ret_pose and pose:
                        for marker_id in marcadores_a_renderizar:
                            if marker_id in marcadores_actuales and marker_id in pose:
                                if marker_id not in escenas:
                                    modelo = crear_modelo_por_id(marker_id)
                                    escenas[marker_id] = crear_escena(modelo, cameraMatrix,
                                                                    int(frame.shape[1]), int(frame.shape[0]))
                                M = from_opencv_to_pygfx(pose[marker_id][0], pose[marker_id][1])
                                escenas[marker_id].actualizar_camara(M)
                                imagen_render = escenas[marker_id].render()
                                imagen_render_bgr = cv2.cvtColor(imagen_render, cv2.COLOR_RGBA2BGRA)
                                frame_visual = cuia.alphaBlending(imagen_render_bgr, frame_visual)

                # --- 5. Actualizar juego con marcadores detectados ---
                if state.gestor_juegos and state.gestor_juegos.juego_activo:
                    state.gestor_juegos.actualizar_marcadores_detectados(marcadores_actuales)

                # --- 6. Estado de escucha de voz ---
                esperando_voz = False
                juego_actual = getattr(state.gestor_juegos, 'juego_activo', None)
                
                if juego_actual and hasattr(juego_actual, 'debe_escuchar_voz'):
                    esperando_voz = juego_actual.debe_escuchar_voz()

                # --- 7. Interfaz especifica para JuegoDescubreAR ---
                if isinstance(juego_actual, JuegoDescubreAR):
                    # La clase maneja toda la logica internamente
                    # Solo mostramos la interfaz visual basica
                    
                    if not juego_actual.fase_escaneo_completada:
                        # Fase de escaneo - Solo mostrar tiempo y contador
                        tiempo_transcurrido = time.time() - juego_actual.tiempo_escaneo if juego_actual.tiempo_escaneo else 0
                        tiempo_restante = max(0, 10 - int(tiempo_transcurrido))
                        
                        draw_text_with_background(frame_visual, f"ESCANEO: {tiempo_restante}s", 
                                                (50, frame_visual.shape[0]-180),
                                                color=(255, 255, 0), bg_color=(100, 100, 0))
                        
                        draw_text_with_background(frame_visual, f"Marcadores: {len(juego_actual.marcadores_detectados_inicial)}", 
                                                (50, frame_visual.shape[0]-150),
                                                color=(0, 255, 255), bg_color=(0, 100, 100))
                    
                    elif not juego_actual.juego_terminado:
                        # Fase de juego - Mostrar pregunta y progreso
                        progreso = len(juego_actual.marcadores_detectados_inicial) - len(juego_actual.marcadores_pendientes) + 1
                        total = len(juego_actual.marcadores_detectados_inicial)
                        
                        if juego_actual.pregunta_actual:
                            draw_text_with_background(frame_visual, juego_actual.pregunta_actual, 
                                                    (50, frame_visual.shape[0]-180),
                                                    color=(0, 255, 255), bg_color=(100, 0, 100))
                        
                        if juego_actual.esperando_nombre:
                            tiempo_esperando = time.time() - juego_actual.tiempo_pregunta if juego_actual.tiempo_pregunta else 0
                            tiempo_restante = max(0, juego_actual.timeout_respuesta - int(tiempo_esperando))
                            
                            draw_text_with_background(frame_visual, f"[MIC] Responde ({tiempo_restante}s)", 
                                                    (50, frame_visual.shape[0]-150),
                                                    color=(255, 255, 255), bg_color=(50, 50, 50))
                        
                        # Mostrar progreso
                        draw_text_with_background(frame_visual, f"Progreso: {progreso}/{total}", 
                                                (50, frame_visual.shape[0]-120),
                                                color=(255, 165, 0), bg_color=(100, 50, 0))
                    
                    elif juego_actual.juego_terminado:
                        # Resultado final - Solo mostrar puntuacion
                        if hasattr(juego_actual, 'resultado_final') and juego_actual.resultado_final:
                            resultado = juego_actual.resultado_final
                            puntuacion = resultado.get("puntuacion", 0)
                            total = resultado.get("total", 0)
                            precision = resultado.get("precision", 0)
                            
                            draw_text_with_background(frame_visual, f"RESULTADO: {puntuacion}/{total} ({precision:.1f}%)", 
                                                    (50, frame_visual.shape[0]-150),
                                                    color=(0, 255, 255), bg_color=(0, 100, 100))

                # --- 8. Interfaces para otros juegos (codigo existente sin cambios) ---
                elif isinstance(juego_actual, JuegoCategoriasAR):
                    if not juego_actual.fase_escaneo_completada:
                        # Fase de escaneo - Solo mostrar tiempo y contador
                        tiempo_transcurrido = time.time() - juego_actual.tiempo_escaneo if juego_actual.tiempo_escaneo else 0
                        tiempo_restante = max(0, 12 - int(tiempo_transcurrido))
                        
                        draw_text_with_background(frame_visual, f"ESCANEO: {tiempo_restante}s", 
                                                (50, frame_visual.shape[0]-180),
                                                color=(255, 255, 0), bg_color=(100, 100, 0))
                        
                        draw_text_with_background(frame_visual, f"Items: {len(juego_actual.marcadores_detectados_inicial)}/6", 
                                                (50, frame_visual.shape[0]-150),
                                                color=(0, 255, 255), bg_color=(0, 100, 100))
                    
                    elif not juego_actual.esperando_respuesta and not juego_actual.juego_terminado:
                        # Fase de colocar elementos - Solo mostrar contador
                        elementos_presentes = [m for m in marcadores_actuales if m in juego_actual.elementos_juego]
                        
                        draw_text_with_background(frame_visual, f"COLOCA ELEMENTOS: {len(elementos_presentes)}/{len(juego_actual.elementos_juego)}", 
                                                (50, frame_visual.shape[0]-150),
                                                color=(255, 165, 0), bg_color=(100, 50, 0))
                    
                    elif juego_actual.esperando_respuesta and not juego_actual.juego_terminado:
                        # Fase de preguntas - Solo mostrar categoria, tiempo y progreso
                        tiempo_esperando = time.time() - juego_actual.tiempo_pregunta if juego_actual.tiempo_pregunta else 0
                        tiempo_restante = max(0, juego_actual.timeout_respuesta - int(tiempo_esperando))
                        
                        if juego_actual.categoria_actual == "frutas":
                            tipo_icono = "[FRUTA]"
                            categoria_nombre = "FRUTAS"
                            respuestas_actuales = len(juego_actual.respuestas_frutas)
                            total_categoria = len(juego_actual.frutas_correctas)
                        else:
                            tipo_icono = "[VERDURA]"
                            categoria_nombre = "VERDURAS"
                            respuestas_actuales = len(juego_actual.respuestas_verduras)
                            total_categoria = len(juego_actual.verduras_correctas)
                        
                        draw_text_with_background(frame_visual, f"{tipo_icono} {categoria_nombre} - {tiempo_restante}s", 
                                                (50, frame_visual.shape[0]-180),
                                                color=(255, 255, 0), bg_color=(100, 100, 0))
                        
                        draw_text_with_background(frame_visual, f"Encontradas: {respuestas_actuales}/{total_categoria}", 
                                                (50, frame_visual.shape[0]-150),
                                                color=(0, 255, 255), bg_color=(0, 100, 100))
                    
                    elif juego_actual.juego_terminado:
                        # Resultado final - Solo mostrar puntuacion
                        if hasattr(juego_actual, 'resultado_final') and juego_actual.resultado_final:
                            resultado = juego_actual.resultado_final
                            total_correctas = resultado.get("total_correctas", 0)
                            total_elementos = resultado.get("total_elementos", 0)
                            porcentaje = resultado.get("porcentaje", 0)
                            
                            draw_text_with_background(frame_visual, f"RESULTADO: {total_correctas}/{total_elementos} ({porcentaje:.1f}%)", 
                                                    (50, frame_visual.shape[0]-150),
                                                    color=(0, 255, 255), bg_color=(0, 100, 100))

                # Activar sistema de escucha si corresponde
                state.esperando_voz = esperando_voz

                # --- 9. Dibujar interfaz del juego ---
                if state.gestor_juegos:
                    state.gestor_juegos.dibujar_interfaz(frame_visual)

                # --- 10. Mostrar estado de escucha activo ---
                if state.esperando_voz:
                    draw_text_with_background(frame_visual, "[VOZ] VOZ ACTIVA", 
                                            (frame_visual.shape[1] - 200, 30), font_scale=0.5,
                                            color=(0, 255, 0), bg_color=(0, 100, 0))

                # --- 11. Guardar puntuacion para JuegoDescubreAR ---
                if isinstance(juego_actual, JuegoDescubreAR) and juego_actual.juego_terminado:
                    if not getattr(juego_actual, "puntuacion_guardada", False):
                        modo = state.gestor_juegos.modo_actual
                        nombre_juego = juego_actual.obtener_nombre()
                        
                        if hasattr(juego_actual, 'resultado_final') and juego_actual.resultado_final:
                            puntuacion = int(juego_actual.resultado_final.get('precision', 0))
                        else:
                            puntuacion_raw = getattr(juego_actual, 'puntuacion', 0)
                            intentos = getattr(juego_actual, 'intentos', 1)
                            puntuacion = int((puntuacion_raw / intentos) * 100) if intentos > 0 else 0
                        
                        exito = guardar_puntuacion_juego(nombre, modo, nombre_juego, puntuacion)
                        if exito:
                            juego_actual.puntuacion_guardada = True
                            print(f"[OK] Puntuacion guardada: {puntuacion}% para {nombre_juego}")

                # --- 12. Guardar puntuacion para JuegoCategoriasAR ---
                if isinstance(juego_actual, JuegoCategoriasAR) and juego_actual.juego_terminado:
                    if not getattr(juego_actual, "puntuacion_guardada", False):
                        modo = state.gestor_juegos.modo_actual
                        nombre_juego = juego_actual.obtener_nombre()
                        
                        if hasattr(juego_actual, 'resultado_final') and juego_actual.resultado_final:
                            puntuacion = int(juego_actual.resultado_final.get('porcentaje', 0))
                        else:
                            puntuacion_raw = getattr(juego_actual, 'puntuacion', 0)
                            intentos = getattr(juego_actual, 'intentos', 1)
                            puntuacion = int((puntuacion_raw / intentos) * 100) if intentos > 0 else 0
                        
                        exito = guardar_puntuacion_juego(nombre, modo, nombre_juego, puntuacion)
                        if exito:
                            juego_actual.puntuacion_guardada = True
                            print(f"[OK] Puntuacion guardada: {puntuacion}% para {nombre_juego}")
                
                # --- 13. Guardar puntuacion para JuegoEncuentraFrutasAR ---
                if isinstance(juego_actual, JuegoEncuentraFrutasAR) and juego_actual.juego_terminado:
                    if not getattr(juego_actual, "puntuacion_guardada", False):
                        modo = state.gestor_juegos.modo_actual
                        nombre_juego = juego_actual.obtener_nombre()
                        
                        if hasattr(juego_actual, 'resultado_final') and juego_actual.resultado_final:
                            puntuacion = int(juego_actual.resultado_final.get('precision', 0))
                        else:
                            puntuacion_raw = getattr(juego_actual, 'puntuacion', 0)
                            intentos = getattr(juego_actual, 'intentos', 1)
                            puntuacion = int((puntuacion_raw / intentos) * 100) if intentos > 0 else 0
                        
                        exito = guardar_puntuacion_juego(nombre, modo, nombre_juego, puntuacion)
                        if exito:
                            juego_actual.puntuacion_guardada = True
                            print(f"[OK] Puntuacion guardada: {puntuacion}% para {nombre_juego}")

                # --- 14. Guardar puntuacion para JuegoMemoriaAR ---
                if isinstance(juego_actual, JuegoMemoriaAR) and juego_actual.juego_terminado:
                    if not getattr(juego_actual, "puntuacion_guardada", False):
                        modo = state.gestor_juegos.modo_actual
                        nombre_juego = juego_actual.obtener_nombre()
                        
                        if hasattr(juego_actual, 'resultado_final') and juego_actual.resultado_final:
                            puntuacion = int(juego_actual.resultado_final.get('porcentaje', 0))
                        else:
                            puntuacion_raw = getattr(juego_actual, 'puntuacion', 0)
                            intentos = getattr(juego_actual, 'intentos', 1)
                            puntuacion = int((puntuacion_raw / intentos) * 100) if intentos > 0 else 0
                        
                        exito = guardar_puntuacion_juego(nombre, modo, nombre_juego, puntuacion)
                        if exito:
                            juego_actual.puntuacion_guardada = True
                            print(f"[OK] Puntuacion guardada: {puntuacion}% para {nombre_juego}")

                # --- 15. Instrucciones generales ---
                instrucciones = []
                
                if isinstance(juego_actual, JuegoDescubreAR):
                    if not juego_actual.fase_escaneo_completada:
                        instrucciones.append("Muestra marcadores uno por uno")
                    elif not juego_actual.juego_terminado:
                        if juego_actual.marcador_actual:
                            instrucciones.append(f"Busca marcador ID: {juego_actual.marcador_actual}")
                        if juego_actual.esperando_nombre:
                            instrucciones.append("Di el nombre en voz alta")
                    elif juego_actual.juego_terminado:
                        instrucciones.append("'otra vez' para repetir | 'salir' para menu")
                
                elif isinstance(juego_actual, JuegoCategoriasAR):
                    if not juego_actual.fase_escaneo_completada:
                        instrucciones.append("Muestra 6 marcadores diferentes")
                    elif not juego_actual.esperando_respuesta and not juego_actual.juego_terminado:
                        instrucciones.append("Coloca todos los elementos")
                    elif juego_actual.esperando_respuesta:
                        categoria = juego_actual.categoria_actual
                        instrucciones.append(f"Di todas las {categoria}")
                        instrucciones.append("'siguiente' para cambiar | 'listo' para terminar")
                    elif juego_actual.juego_terminado:
                        instrucciones.append("'otra vez' para repetir | 'salir' para menu")
                
                # Mostrar instrucciones
                for i, instruccion in enumerate(instrucciones):
                    y_offset = 80 + (i * 25)
                    draw_text_with_background(frame_visual, instruccion, 
                                            (50, frame_visual.shape[0] - y_offset), font_scale=0.5,
                                            color=(200, 200, 200), bg_color=(50, 50, 50))

                # Mostrar frame final
                frame = frame_visual
            
            cv2.imshow("Kids&Veggies - AR Learning Game", frame)
            
            if state.fase == "salir" or cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("\n🛑 Aplicación interrumpida por el usuario")
    
    finally:
        voice_thread_active = False
        ar.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()