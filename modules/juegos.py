import modules.cuia as cuia
import numpy as np
import random
import time
import cv2
from models.modelos import MODELOS_FRUTAS_VERDURAS, crear_modelo_por_id, obtener_info_modelo

class GestorJuegosAR:
    """Gestor de juegos adaptado para la aplicación AR"""
    
    def __init__(self):
        self.juego_activo = None
        self.estado_juego = "menu_juegos"  # menu_juegos, en_juego, pausa, resultados
        self.modo_actual = None  # "entrenamiento" o "evaluacion"
        self.tipo_juego_actual = None
        self.mensajes_pantalla = []
        self.modelos_activos = []
        self.esperando_respuesta = False
        self.respuesta_usuario = ""
        self.tiempo_ultima_accion = time.time()
        self.puntuacion_guardada = False  
        
        # Juegos disponibles por modo
        self.juegos_entrenamiento = {
            "descubre": {
                "nombre": "Descubre y Nombra",
                "descripcion": "Aprende el nombre de frutas y verduras",
                "comando": "descubre"
            },
            "frutas": {
                "nombre": "Encuentra las Frutas", 
                "descripcion": "Identifica solo las frutas",
                "comando": "frutas"
            }
        }
        
        self.juegos_evaluacion = {
            "categorias": {
                "nombre": "Agrupa por Categorias",
                "descripcion": "Separa frutas de verduras", 
                "comando": "categorias"
            },
            "memoria": {
                "nombre": "Juego de Memoria",
                "descripcion": "Recuerda la secuencia",
                "comando": "memoria"
            }
        }
        
    def establecer_modo(self, modo):
        """Establece el modo de juego (entrenamiento o evaluacion)"""
        self.modo_actual = modo
        self.estado_juego = "menu_juegos"
        self.mensajes_pantalla = [f"Modo {modo.upper()} seleccionado"]
        self.puntuacion_guardada = False 
        
    def obtener_juegos_disponibles(self):
        """Devuelve los juegos disponibles según el modo actual"""
        if self.modo_actual == "entrenamiento":
            return self.juegos_entrenamiento
        elif self.modo_actual == "evaluacion":
            return self.juegos_evaluacion
        return {}
        
    def iniciar_juego(self, tipo_juego):
        """Inicia un juego específico"""
        self.tipo_juego_actual = tipo_juego
        self.estado_juego = "en_juego"
        self.modelos_activos = []
        self.mensajes_pantalla = []
        self.tiempo_ultima_accion = time.time()
        self.puntuacion_guardada = False 
        
        if tipo_juego == "descubre":
            self.juego_activo = JuegoDescubreAR()
        elif tipo_juego == "frutas":
            self.juego_activo = JuegoEncuentraFrutasAR()
        elif tipo_juego == "categorias":
            self.juego_activo = JuegoCategoriasAR()
        elif tipo_juego == "memoria":
            self.juego_activo = JuegoMemoriaAR()
        else:
            self.mensajes_pantalla = ["Juego no encontrado"]
            self.estado_juego = "menu_juegos"
            return
            
        # Inicializar el juego
        resultado = self.juego_activo.iniciar()
        self.procesar_resultado_juego(resultado)
        
    def procesar_comando_voz(self, comando):
        """Procesa comandos de voz según el estado actual"""
        comando = comando.lower().strip()
        
        if self.estado_juego == "menu_juegos":
            juegos = self.obtener_juegos_disponibles()
            for key, juego in juegos.items():
                if comando in juego["comando"]:
                    self.iniciar_juego(key)
                    return True
                    
            if "volver" in comando or "menu" in comando:
                return "volver_modo"
                
        elif self.estado_juego == "en_juego" and self.juego_activo:
            resultado = self.juego_activo.procesar_comando(comando)
            self.procesar_resultado_juego(resultado)
            return True
            
        elif self.estado_juego == "resultados":
            if "continuar" in comando or "siguiente" in comando:
                if self.juego_activo and hasattr(self.juego_activo, 'continuar'):
                    resultado = self.juego_activo.continuar()
                    self.procesar_resultado_juego(resultado)
                else:
                    self.estado_juego = "menu_juegos"
                    self.puntuacion_guardada = False  
                return True
            elif "menu" in comando or "volver" in comando:
                self.estado_juego = "menu_juegos"
                self.puntuacion_guardada = False  
                return True
                
        return False
        
    def procesar_resultado_juego(self, resultado):
        """Procesa el resultado devuelto por un juego"""
        if not resultado:
            return
            
        if "mensajes" in resultado:
            self.mensajes_pantalla = resultado["mensajes"]
            
        if "modelos" in resultado:
            self.modelos_activos = resultado["modelos"]
            
        if "estado" in resultado:
            if resultado["estado"] == "terminado":
                self.estado_juego = "resultados"
                self.puntuacion_guardada = False 
            elif resultado["estado"] == "esperando":
                self.esperando_respuesta = True
            elif resultado["estado"] == "menu":
                self.estado_juego = "menu_juegos"
                self.puntuacion_guardada = False  
    
    def juego_terminado(self):
        """Verifica si el juego actual ha terminado"""
        return self.estado_juego == "resultados"
    
    def debe_guardar_puntuacion(self):
        """Verifica si se debe guardar la puntuación (juego terminado y no guardado aún)"""
        return self.juego_terminado() and not self.puntuacion_guardada
    
    def marcar_puntuacion_guardada(self):
        """Marca la puntuación como guardada"""
        self.puntuacion_guardada = True
    
    def obtener_datos_juego_terminado(self):
        """Obtiene los datos del juego terminado para guardar"""
        if not self.juego_terminado() or not self.juego_activo:
            return None
            
        datos = {
            "modo": self.modo_actual,
            "tipo_juego": self.tipo_juego_actual,
            "timestamp": time.time()
        }
        
        # Agregar puntuación si existe
        if hasattr(self.juego_activo, 'obtener_puntuacion'):
            datos["puntuacion"] = self.juego_activo.obtener_puntuacion()
            
        # Agregar otros datos del juego si existen
        if hasattr(self.juego_activo, 'obtener_estadisticas'):
            datos["estadisticas"] = self.juego_activo.obtener_estadisticas()
            
        return datos
                
    def actualizar_marcadores_detectados(self, marcadores_detectados):
        """Actualiza los marcadores detectados en el juego actual"""
        if self.juego_activo and hasattr(self.juego_activo, 'actualizar_marcadores'):
            resultado = self.juego_activo.actualizar_marcadores(marcadores_detectados)
            self.procesar_resultado_juego(resultado)
            
    def dibujar_interfaz(self, frame):
        """Dibuja la interfaz del sistema de juegos en el frame"""
        alto, ancho = frame.shape[:2]
        
        if self.estado_juego == "menu_juegos":
            self._dibujar_menu_juegos(frame)
        elif self.estado_juego == "en_juego":
            self._dibujar_juego_activo(frame)
        elif self.estado_juego == "resultados":
            self._dibujar_resultados(frame)
            
        # Siempre mostrar información del modo actual
        if self.modo_actual:
            self._draw_text_with_background(frame, f"MODO: {self.modo_actual.upper()}", (10, 30),
                                          font_scale=0.8, color=(255, 255, 255), bg_color=(100, 0, 100))
            
    def _dibujar_menu_juegos(self, frame):
        """Dibuja el menú de selección de juegos"""
        y_pos = 80
        self._draw_text_with_background(frame, "SELECCIONA UN JUEGO:", (50, y_pos),
                                      color=(255, 255, 255), bg_color=(0, 100, 0))
        y_pos += 50
        
        juegos = self.obtener_juegos_disponibles()
        for key, juego in juegos.items():
            self._draw_text_with_background(frame, f"Di '{juego['comando']}' - {juego['nombre']}", 
                                          (50, y_pos), font_scale=0.7,
                                          color=(255, 255, 255), bg_color=(0, 0, 100))
            y_pos += 30
            self._draw_text_with_background(frame, f"  {juego['descripcion']}", 
                                          (70, y_pos), font_scale=0.5,
                                          color=(200, 200, 200), bg_color=(50, 50, 50))
            y_pos += 40
            
        self._draw_text_with_background(frame, "Di 'volver' para cambiar modo", (50, y_pos),
                                      font_scale=0.6, color=(255, 255, 0), bg_color=(100, 100, 0))
        
    def _dibujar_juego_activo(self, frame):
        """Dibuja la interfaz del juego activo"""
        if not self.juego_activo:
            return
            
        # Título del juego
        nombre_juego = self.juego_activo.obtener_nombre()
        self._draw_text_with_background(frame, nombre_juego, (50, 80),
                                      font_scale=1.0, color=(255, 255, 255), bg_color=(0, 100, 0))
        
        # Puntuación si está disponible
        if hasattr(self.juego_activo, 'obtener_puntuacion'):
            puntuacion = self.juego_activo.obtener_puntuacion()
            self._draw_text_with_background(frame, f"Puntuacion: {puntuacion}", (50, 120),
                                          font_scale=0.7, color=(255, 255, 0), bg_color=(100, 100, 0))
        
        # Mensajes del juego
        y_pos = 160
        for mensaje in self.mensajes_pantalla:
            self._draw_text_with_background(frame, mensaje, (50, y_pos),
                                          font_scale=0.7, color=(255, 255, 255), bg_color=(0, 50, 100))
            y_pos += 35
            
    def _dibujar_resultados(self, frame):
        """Dibuja los resultados del juego"""
        if not self.juego_activo:
            return
            
        alto, ancho = frame.shape[:2]
        
        self._draw_text_with_background(frame, "¡JUEGO TERMINADO!", (50, 80),
                                      font_scale=1.2, color=(255, 255, 255), bg_color=(0, 150, 0))
        
        # Mostrar resultados
        if hasattr(self.juego_activo, 'obtener_resultados'):
            resultados = self.juego_activo.obtener_resultados()
            y_pos = 130
            for resultado in resultados:
                self._draw_text_with_background(frame, resultado, (50, y_pos),
                                              font_scale=0.8, color=(255, 255, 255), bg_color=(0, 100, 0))
                y_pos += 40
        
    def _draw_text_with_background(self, frame, text, position, font_scale=0.8, color=(255, 255, 255), bg_color=(0, 0, 0)):
        """Dibuja texto con fondo - función auxiliar"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        # Obtener tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Dibujar fondo
        x, y = position
        cv2.rectangle(frame, (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), bg_color, -1)
        
        # Dibujar texto
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    def obtener_marcadores_para_renderizar(self):
        """
        Retorna los marcadores que el juego actual quiere que se rendericen
        """
        if self.juego_activo:
            # Si el juego tiene un método específico para esto
            if hasattr(self.juego_activo, 'obtener_marcadores_renderizado'):
                return self.juego_activo.obtener_marcadores_renderizado()
            
            # Para el juego Descubre, mostrar el marcador actual durante el juego
            if hasattr(self.juego_activo, 'marcador_actual') and self.juego_activo.marcador_actual:
                return [self.juego_activo.marcador_actual]
            
            # Durante escaneo inicial, mostrar todos los marcadores detectados
            if hasattr(self.juego_activo, 'marcadores_detectados_inicial'):
                return list(self.juego_activo.marcadores_detectados_inicial)
        
        return []
    
    def debe_escuchar_voz(self):
        """Determina si debe escuchar comandos de voz"""
        return (self.esperando_respuesta and not self.juego_terminado()) or self.juego_terminado()


class JuegoBaseAR:
    """Clase base para todos los juegos AR"""
    
    def __init__(self):
        self.activo = False
        self.puntuacion = 0
        self.intentos = 0
        self.fase_actual = "inicio"
        self.marcadores_requeridos = []
        self.marcadores_detectados = []
        self.porcentaje = 0
        
    def iniciar(self):
        """Inicia el juego"""
        self.activo = True
        return self._inicializar_juego()
        
    def _inicializar_juego(self):
        """Método a implementar por cada juego"""
        pass
        
    def procesar_comando(self, comando):
        """Procesa comandos de voz"""
        return None
        
    def actualizar_marcadores(self, marcadores):
        """Actualiza marcadores detectados"""
        self.marcadores_detectados = marcadores
        return self._procesar_marcadores()
        
    def _procesar_marcadores(self):
        """Procesa los marcadores detectados"""
        return None
        
    def obtener_nombre(self):
        """Devuelve el nombre del juego"""
        return "Juego Base"
        
    def obtener_puntuacion(self):
        """Devuelve la puntuación actual"""
        if self.intentos > 0:
            self.porcentaje = (self.puntuacion / self.intentos) * 100
            return f"{self.puntuacion}/{self.intentos} ({self.porcentaje:.1f}%)"
        return "0/0"
        
    def obtener_resultados(self):
        """Devuelve lista de strings con los resultados"""
        return [f"Puntuacion final: {self.obtener_puntuacion()}"]

    def debe_escuchar_voz(self):
        """Indica si el juego actual espera una respuesta por voz"""
        return False

class JuegoMemoriaAR(JuegoBaseAR):
    """Juego de Memoria para AR"""
    
    def __init__(self):
        super().__init__()
        # Variables de control del escaneo inicial
        self.marcadores_detectados_inicial = set()
        self.tiempo_escaneo = None
        self.fase_escaneo_completada = False
        
        # Variables de la secuencia de memoria
        self.secuencia_memoria = []
        self.marcador_actual_mostrando = None
        self.indice_secuencia = 0
        self.tiempo_mostrar_elemento = None
        # 3 segundos por elemento
        self.duracion_elemento = 3  
        self.fase_mostrando_secuencia = False
        self.secuencia_completada = False
        
        # Variables para la respuesta del usuario
        self.esperando_respuesta = False
        self.respuesta_usuario = []
        self.tiempo_pregunta = None
        # 30 segundos para responder
        self.timeout_respuesta = 30  
        
        # Control de fin de juego
        self.juego_terminado = False
        self.resultado_final = None
        
    def obtener_nombre(self):
        return "MEMORIA AR"
        
    def _inicializar_juego(self):
        """Inicia el juego con escaneo inicial"""
        self.marcadores_detectados_inicial.clear()
        self.tiempo_escaneo = time.time()
        self.fase_escaneo_completada = False
        self.secuencia_memoria = []
        self.marcador_actual_mostrando = None
        self.indice_secuencia = 0
        self.tiempo_mostrar_elemento = None
        self.fase_mostrando_secuencia = False
        self.secuencia_completada = False
        self.esperando_respuesta = False
        self.respuesta_usuario = []
        self.tiempo_pregunta = None
        self.juego_terminado = False
        self.resultado_final = None
        
        return {
            "estado": "escaneo_inicial",
            "mensajes": [
                "ESCANEO INICIAL (10 segundos)",
                "Muestra exactamente 3 marcadores diferentes",
                "Colocalos uno por uno frente a la camara"
            ],
            "esperando_marcadores": list(MODELOS_FRUTAS_VERDURAS.keys()) if 'MODELOS_FRUTAS_VERDURAS' in globals() else [],
        }
        
    def actualizar_marcadores(self, marcadores_visibles):
        """Actualiza según los marcadores detectados"""
        current_time = time.time()
        
        # FASE 1: ESCANEO INICIAL (primeros 10 segundos)
        if not self.fase_escaneo_completada:
            tiempo_transcurrido = current_time - self.tiempo_escaneo
            tiempo_restante = max(0, 10 - int(tiempo_transcurrido))
            
            # Añadir marcadores válidos detectados
            for marker_id in marcadores_visibles:
                if marker_id in MODELOS_FRUTAS_VERDURAS:
                    self.marcadores_detectados_inicial.add(marker_id)
            
            mensajes = [
                f"ESCANEO INICIAL - {tiempo_restante}s restantes",
                f"Marcadores encontrados: {len(self.marcadores_detectados_inicial)}/3"
            ]
            
            # Mostrar qué marcadores se han encontrado
            for marker_id in sorted(self.marcadores_detectados_inicial):
                info = obtener_info_modelo(marker_id)
                mensajes.append(f"{info['nombre']}")
            
            if tiempo_transcurrido >= 10:
                if len(self.marcadores_detectados_inicial) >= 3:
                    # Completar escaneo e iniciar secuencia
                    self.fase_escaneo_completada = True
                    # Tomar solo los primeros 3 marcadores
                    marcadores_lista = list(self.marcadores_detectados_inicial)[:3]
                    # Crear secuencia aleatoria
                    self.secuencia_memoria = marcadores_lista.copy()
                    random.shuffle(self.secuencia_memoria)
                    
                    return self._iniciar_secuencia()
                else:
                    mensajes.append(f" Necesitas exactamente 3 marcadores (tienes {len(self.marcadores_detectados_inicial)})")
                    # Reiniciar escaneo si no hay suficientes marcadores
                    if tiempo_transcurrido >= 15:
                        self.tiempo_escaneo = current_time
                        self.marcadores_detectados_inicial.clear()
            
            return {
                "estado": "escaneo_inicial",
                "mensajes": mensajes,
                "esperando_marcadores": list(MODELOS_FRUTAS_VERDURAS.keys()),
                "modelos": [crear_modelo_por_id(mid) for mid in marcadores_visibles if mid in MODELOS_FRUTAS_VERDURAS]
            }
        
        # FASE 2: MOSTRANDO SECUENCIA
        elif self.fase_mostrando_secuencia and not self.secuencia_completada:
            return self._actualizar_secuencia(current_time)
        
        # FASE 3: ESPERANDO RESPUESTA DEL USUARIO
        elif self.esperando_respuesta:
            return self._manejar_fase_respuesta(current_time)
        
        # FASE 4: JUEGO TERMINADO
        elif self.juego_terminado:
            return self._mostrar_resultado_final()
        
        return {"estado": "error", "mensajes": ["Estado no reconocido"]}
    
    def _iniciar_secuencia(self):
        """Inicia la fase de mostrar la secuencia"""
        self.fase_mostrando_secuencia = True
        self.indice_secuencia = 0
        self.tiempo_mostrar_elemento = time.time()
        self.marcador_actual_mostrando = self.secuencia_memoria[0]
        
        info = obtener_info_modelo(self.marcador_actual_mostrando)
        
        return {
            "estado": "mostrando_secuencia",
            "mensajes": [
                " MEMORIZA LA SECUENCIA",
                f"Elemento 1/3: {info['nombre']}",
                "Observa el orden..."
            ],
            "esperando_marcadores": [self.marcador_actual_mostrando],
            "modelos": [crear_modelo_por_id(self.marcador_actual_mostrando)]
        }
    
    def _actualizar_secuencia(self, current_time):
        """Actualiza la visualización de la secuencia"""
        tiempo_elemento = current_time - self.tiempo_mostrar_elemento
        
        if tiempo_elemento >= self.duracion_elemento:
            # Pasar al siguiente elemento
            self.indice_secuencia += 1
            
            if self.indice_secuencia >= len(self.secuencia_memoria):
                # Secuencia completada
                self.secuencia_completada = True
                self.fase_mostrando_secuencia = False
                return self._iniciar_fase_respuesta()
            else:
                # Mostrar siguiente elemento
                self.tiempo_mostrar_elemento = current_time
                self.marcador_actual_mostrando = self.secuencia_memoria[self.indice_secuencia]
                
                info = obtener_info_modelo(self.marcador_actual_mostrando)
                
                return {
                    "estado": "mostrando_secuencia",
                    "mensajes": [
                        " MEMORIZA LA SECUENCIA",
                        f"Elemento {self.indice_secuencia + 1}/3: {info['nombre']}",
                        "Observa el orden..."
                    ],
                    "esperando_marcadores": [self.marcador_actual_mostrando],
                    "modelos": [crear_modelo_por_id(self.marcador_actual_mostrando)]
                }
        else:
            # Continuar mostrando el elemento actual
            info = obtener_info_modelo(self.marcador_actual_mostrando)
            tiempo_restante = self.duracion_elemento - int(tiempo_elemento)
            
            return {
                "estado": "mostrando_secuencia",
                "mensajes": [
                    " MEMORIZA LA SECUENCIA",
                    f"Elemento {self.indice_secuencia + 1}/3: {info['nombre']}",
                    f"Siguiente en {tiempo_restante}s..."
                ],
                "esperando_marcadores": [self.marcador_actual_mostrando],
                "modelos": [crear_modelo_por_id(self.marcador_actual_mostrando)]
            }
    
    def _iniciar_fase_respuesta(self):
        """Inicia la fase donde el usuario debe decir la secuencia"""
        self.esperando_respuesta = True
        self.tiempo_pregunta = time.time()
        self.respuesta_usuario = []
        
        # Crear mensaje con la secuencia para referencia
        nombres_secuencia = [obtener_info_modelo(mid)['nombre'] for mid in self.secuencia_memoria]
        
        return {
            "estado": "esperando_respuesta",
            "mensajes": [
                " AHORA REPITE LA SECUENCIA",
                "Di los nombres EN ORDEN, uno por uno",
                "Ejemplo: 'pera', 'limón', 'lechuga'",
                f"Tienes {self.timeout_respuesta} segundos"
            ],
            "esperando_marcadores": [],
            "modelos": []
        }
    
    def _manejar_fase_respuesta(self, current_time):
        """Maneja la fase de respuesta del usuario"""
        if not self.esperando_respuesta:
            # La respuesta ya se completó, mostrar resultado
            return self._mostrar_resultado_final()
        
        tiempo_esperando = current_time - self.tiempo_pregunta
        tiempo_restante = max(0, self.timeout_respuesta - int(tiempo_esperando))
        
        if tiempo_esperando >= self.timeout_respuesta:
            # Timeout - mostrar resultado
            return self._timeout_respuesta()
        
        progreso_respuesta = len(self.respuesta_usuario)
        mensajes = [
            "REPITE LA SECUENCIA",
            f"Elementos dados: {progreso_respuesta}/3",
            f"Tiempo restante: {tiempo_restante}s"
        ]
        
        # Mostrar respuestas dadas hasta ahora
        if self.respuesta_usuario:
            mensajes.append("Has dicho:")
            for i, respuesta in enumerate(self.respuesta_usuario):
                mensajes.append(f"{i+1}. {respuesta}")
        
        return {
            "estado": "escuchando_respuesta",
            "mensajes": mensajes,
            "esperando_marcadores": [],
            "modelos": []
        }
    
    def _timeout_respuesta(self):
        """Maneja el timeout de respuesta"""
        nombres_correctos = [obtener_info_modelo(mid)['nombre'] for mid in self.secuencia_memoria]
        
        self.juego_terminado = True
        self.resultado_final = {
            "correcto": False,
            "razon": "timeout",
            "secuencia_correcta": nombres_correctos,
            "tu_respuesta": self.respuesta_usuario
        }
        
        return self._mostrar_resultado_final()
    
    def _mostrar_resultado_final(self):
        """Muestra el resultado final del juego"""
        if not self.resultado_final:
            return {"estado": "error", "mensajes": ["Error: No hay resultado"]}
        
        mensajes = []
        
        if self.resultado_final["correcto"]:
            mensajes.extend([
                "¡CORRECTO!",
                "¡Tienes una excelente memoria!",
                f"Puntuacion: {self.puntuacion}/1"
            ])
        else:
            if self.resultado_final["razon"] == "timeout":
                mensajes.append("Se acabo el tiempo")
            else:
                mensajes.append("Secuencia incorrecta")
            
            mensajes.extend([
                "Secuencia correcta:",
                " ".join(self.resultado_final["secuencia_correcta"])
            ])
            
            if self.resultado_final["tu_respuesta"]:
                mensajes.extend([
                    "Tu respuesta:",
                    " ".join(self.resultado_final["tu_respuesta"])
                ])
        
        mensajes.extend([
            "",
            "Di 'otra vez' para jugar de nuevo",
            "Di 'salir' para terminar"
        ])
        
        return {
            "estado": "resultado_final",
            "mensajes": mensajes,
            "esperando_marcadores": [],
            "modelos": []
        }
    
    def verificar_respuesta_elemento(self, texto, respuesta_correcta):
        """Verificar si un elemento de la respuesta es correcto"""
        
        texto = texto.lower().strip()
        respuesta_correcta = respuesta_correcta.lower().strip()
        
        # Verificaciones directas
        if respuesta_correcta in texto:
            return True
        
        # Verificaciones con variaciones y sinónimos
        variaciones = {
            'pera': ['pera'],
            'cebolleta': ['cebolleta', 'cebollino', 'cebolla verde'],
            'cebolla': ['cebolla'],
            'lechuga': ['lechuga'],
            'limon': ['limon', 'limón'],  
            'pimiento rojo': ['pimiento rojo', 'pimiento', 'pimentón rojo', 'chile rojo'],
            'pimiento verde': ['pimiento verde', 'pimiento', 'pimentón verde', 'chile verde'],
            'uvas': ['uvas', 'uva', 'racimo'],
            'zanahoria': ['zanahoria'],
        }
        
        # Buscar variaciones
        if respuesta_correcta in variaciones:
            for variacion in variaciones[respuesta_correcta]:
                if variacion in texto:
                    return True
        
        # Verificación por palabras clave
        palabras_respuesta = respuesta_correcta.split()
        for palabra in palabras_respuesta:
            if len(palabra) > 2 and palabra in texto:
                return True
        
        return False

    def procesar_comando(self, comando):
        """Procesa comandos de voz durante el juego"""
        comando_lower = comando.lower().strip()
        
        # Comandos especiales en resultado final
        if self.juego_terminado:
            if any(palabra in comando_lower for palabra in ['otra vez', 'repetir', 'nuevo', 'again']):
                # Reiniciar juego
                return self._inicializar_juego()
            elif any(palabra in comando_lower for palabra in ['salir', 'terminar', 'exit', 'quit']):
                return {
                    "estado": "salir",
                    "mensajes": [" ¡Gracias por jugar!", "Hasta la proxima"]
                }
            else:
                return None  
        
        # Procesar respuesta durante la fase de escucha
        if self.esperando_respuesta and len(self.respuesta_usuario) < 3:
            # Buscar cuál fruta/verdura menciona
            nombres_correctos = [obtener_info_modelo(mid)['nombre'] for mid in self.secuencia_memoria]
            
            elemento_detectado = None
            for nombre in nombres_correctos:
                if self.verificar_respuesta_elemento(comando, nombre):
                    elemento_detectado = nombre
                    break
            
            if elemento_detectado:
                self.respuesta_usuario.append(elemento_detectado)
                
                # Verificar si ya completó la secuencia
                if len(self.respuesta_usuario) >= 3:
                    self.esperando_respuesta = False
                    return self._evaluar_respuesta_completa()
                else:
                    # Continuar esperando más elementos
                    return None
            else:
                # No se reconoció ningún elemento válido, continuar esperando
                return None
        
        return None
       
    def _evaluar_respuesta_completa(self):
        """Evalúa la respuesta completa del usuario"""
        nombres_correctos = [obtener_info_modelo(mid)['nombre'] for mid in self.secuencia_memoria]
        
        # Verificar si la secuencia es correcta
        es_correcta = len(self.respuesta_usuario) == len(nombres_correctos)
        
        if es_correcta:
            for i in range(len(nombres_correctos)):
                elemento_correcto = self.verificar_respuesta_elemento(
                    self.respuesta_usuario[i], 
                    nombres_correctos[i]
                )
                
                if not elemento_correcto:
                    es_correcta = False
                    break
        
        # Actualizar puntuación
        if es_correcta:
            self.puntuacion += 1
        
        self.intentos += 1
        self.juego_terminado = True
        
        self.esperando_respuesta = False
        
        self.resultado_final = {
            "correcto": es_correcta,
            "razon": "respuesta_completa",
            "secuencia_correcta": nombres_correctos,
            "tu_respuesta": self.respuesta_usuario
        }
        
        return self._mostrar_resultado_final()
    
    def obtener_estado_detallado(self):
        """Información detallada del juego"""
        base_info = super().obtener_estado_detallado()
        base_info.update({
            "marcadores_detectados": len(self.marcadores_detectados_inicial),
            "fase_escaneo_completada": self.fase_escaneo_completada,
            "secuencia_memoria": self.secuencia_memoria,
            "indice_secuencia": self.indice_secuencia,
            "fase_mostrando_secuencia": self.fase_mostrando_secuencia,
            "secuencia_completada": self.secuencia_completada,
            "esperando_respuesta": self.esperando_respuesta,
            "respuesta_usuario": self.respuesta_usuario,
            "juego_terminado": self.juego_terminado,
            "elementos_respondidos": len(self.respuesta_usuario) if self.respuesta_usuario else 0
        })
        return base_info
    
    def obtener_marcadores_renderizado(self):
        """Retorna qué marcadores se deben renderizar en 3D según el estado del juego"""
        if not self.fase_escaneo_completada:
            # Durante escaneo: mostrar todos los marcadores válidos detectados
            return list(self.marcadores_detectados_inicial)
        elif self.fase_mostrando_secuencia and self.marcador_actual_mostrando:
            # Durante secuencia: mostrar solo el marcador actual
            return [self.marcador_actual_mostrando]
        else:
            # Durante respuesta o fin: no mostrar marcadores
            return []
    
    def debe_escuchar_voz(self):
        """Determina si debe escuchar comandos de voz"""
        return (self.esperando_respuesta and not self.juego_terminado) or self.juego_terminado

class JuegoDescubreAR(JuegoBaseAR):
    """Juego Descubre y Nombra para AR"""
    
    def __init__(self):
        super().__init__()
        self.marcadores_detectados_inicial = set()
        self.marcadores_pendientes = []
        self.marcador_actual = None
        self.esperando_nombre = False
        self.tiempo_escaneo = None
        self.fase_escaneo_completada = False
        
        self.pregunta_actual = None
        self.respuesta_correcta = None
        self.tiempo_pregunta = None
        # 15 segundos para responder
        self.timeout_respuesta = 15  
        
        # Control de fin de juego
        self.juego_terminado = False
        self.resultado_final = None
        
    def verificar_respuesta(self, texto, respuesta_correcta):
        """Verificar si la respuesta del usuario es correcta con variaciones"""
        texto = texto.lower().strip()
        respuesta_correcta = respuesta_correcta.lower().strip()
        
        # Verificaciones directas
        if respuesta_correcta in texto:
            return True
        
        # Verificaciones adicionales con variaciones y sinónimos
        variaciones = {
            'pera': ['pera'],
            'cebolleta': ['cebolleta', 'cebollino', 'cebolla verde'],
            'cebolla': ['cebolla'],
            'lechuga': ['lechuga'],
            'limon': ['limon'],
            'pimiento rojo': ['pimiento rojo', 'pimiento', 'pimentón rojo', 'chile rojo'],
            'pimiento verde': ['pimiento verde', 'pimiento', 'pimentón verde', 'chile verde'],
            'uvas': ['uvas', 'uva', 'racimo'],
            'zanahoria': ['zanahoria'],
        }
        
        # Buscar variaciones
        if respuesta_correcta in variaciones:
            for variacion in variaciones[respuesta_correcta]:
                if variacion in texto:
                    return True
        
        # Verificación por palabras clave
        palabras_respuesta = respuesta_correcta.split()
        for palabra in palabras_respuesta:
            if len(palabra) > 2 and palabra in texto:
                return True
        
        return False
    
    def generar_pregunta(self, marker_id):
        """Generar diferentes tipos de preguntas para variedad"""
        if marker_id not in MODELOS_FRUTAS_VERDURAS:
            return None, None
            
        info = obtener_info_modelo(marker_id)
        nombre = info['nombre']
        tipo = info.get('tipo', 'fruta o verdura')
        
        # Diferentes tipos de preguntas para hacer el juego más dinámico
        tipos_pregunta = [
            f"¿Que {tipo} es esta?",
            f"Dime el nombre de esta {tipo}",
            f"¿Como se llama lo que ves?",
            f"Identifica esta {tipo}",
            f"Nombra esta {tipo}"
        ]
        
        pregunta = random.choice(tipos_pregunta)
        return pregunta, nombre
        
    def obtener_nombre(self):
        return "DESCUBRE Y NOMBRA"
        
    def _inicializar_juego(self):
        """Inicia el juego con escaneo inicial"""
        self.marcadores_detectados_inicial.clear()
        self.marcadores_pendientes = []
        self.marcador_actual = None
        self.esperando_nombre = False
        self.tiempo_escaneo = time.time()
        self.fase_escaneo_completada = False
        self.pregunta_actual = None
        self.respuesta_correcta = None
        self.tiempo_pregunta = None
        
        # Resetear control de fin de juego
        self.juego_terminado = False
        self.resultado_final = None
        
        return {
            "estado": "escaneo_inicial",
            "mensajes": [
                "ESCANEO INICIAL (10 segundos)",
                "Muestra todos los marcadores que quieres incluir",
                "Colocalos uno por uno frente a la camara"
            ],
            "esperando_marcadores": list(MODELOS_FRUTAS_VERDURAS.keys()) if 'MODELOS_FRUTAS_VERDURAS' in globals() else [],
        }
        
    def actualizar_marcadores(self, marcadores_visibles):
        """Actualiza según los marcadores detectados"""
        current_time = time.time()
        
        # FASE 1: ESCANEO INICIAL (primeros 10 segundos)
        if not self.fase_escaneo_completada:
            tiempo_transcurrido = current_time - self.tiempo_escaneo
            tiempo_restante = max(0, 10 - int(tiempo_transcurrido))
            
            # Añadir marcadores válidos detectados
            for marker_id in marcadores_visibles:
                if marker_id in MODELOS_FRUTAS_VERDURAS:
                    self.marcadores_detectados_inicial.add(marker_id)
            
            mensajes = [
                f"ESCANEO INICIAL - {tiempo_restante}s restantes",
                f"Marcadores encontrados: {len(self.marcadores_detectados_inicial)}"
            ]
            
            # Mostrar qué marcadores se han encontrado
            for marker_id in sorted(self.marcadores_detectados_inicial):
                info = obtener_info_modelo(marker_id)
                mensajes.append(f"{info['nombre']}")
            
            if tiempo_transcurrido >= 10:
                if self.marcadores_detectados_inicial:
                    # Completar escaneo e iniciar juego
                    self.fase_escaneo_completada = True
                    self.marcadores_pendientes = list(self.marcadores_detectados_inicial)
                    random.shuffle(self.marcadores_pendientes)
                    
                    return self._mostrar_siguiente_marcador()
                else:
                    mensajes.append("No se encontraron marcadores validos")
                    # Reiniciar escaneo
                    if tiempo_transcurrido >= 15:
                        self.tiempo_escaneo = current_time
            
            return {
                "estado": "escaneo_inicial",
                "mensajes": mensajes,
                "esperando_marcadores": list(MODELOS_FRUTAS_VERDURAS.keys()),
                "modelos": [crear_modelo_por_id(mid) for mid in marcadores_visibles if mid in MODELOS_FRUTAS_VERDURAS]
            }
        
        # FASE 2: JUEGO PRINCIPAL
        elif not self.juego_terminado:
            if not self.marcadores_pendientes:
                return self._finalizar_juego()
            
            # Verificar si el marcador actual está visible
            if self.marcador_actual in marcadores_visibles:
                info = obtener_info_modelo(self.marcador_actual)
                progreso = len(self.marcadores_detectados_inicial) - len(self.marcadores_pendientes) + 1
                total = len(self.marcadores_detectados_inicial)
                
                # Generar pregunta si no existe
                if not self.pregunta_actual:
                    self.pregunta_actual, self.respuesta_correcta = self.generar_pregunta(self.marcador_actual)
                    self.tiempo_pregunta = current_time
                    self.esperando_nombre = True
                
                # Verificar timeout de respuesta
                tiempo_esperando = current_time - self.tiempo_pregunta if self.tiempo_pregunta else 0
                tiempo_restante = max(0, self.timeout_respuesta - int(tiempo_esperando))
                
                mensajes = [
                    f"Elemento {progreso}/{total}",
                    self.pregunta_actual or f"¿Que {info['tipo']} es esta?",
                ]
                
                if self.esperando_nombre:
                    mensajes.append(f" Responde ahora ({tiempo_restante}s)")
                    
                    # Si se agota el tiempo, pasar al siguiente
                    if tiempo_esperando >= self.timeout_respuesta:
                        return self._timeout_respuesta()
                else:
                    mensajes.append("Preparando pregunta...")
                
                return {
                    "estado": "preguntando" if not self.esperando_nombre else "escuchando",
                    "mensajes": mensajes,
                    "esperando_marcadores": [self.marcador_actual],
                    "modelos": [crear_modelo_por_id(self.marcador_actual)]
                }
            else:
                # Marcador no visible, mostrar instrucciones
                info = obtener_info_modelo(self.marcador_actual) if self.marcador_actual else None
                progreso = len(self.marcadores_detectados_inicial) - len(self.marcadores_pendientes) + 1
                total = len(self.marcadores_detectados_inicial)
                
                # Resetear pregunta cuando no está visible
                self.pregunta_actual = None
                self.respuesta_correcta = None
                self.tiempo_pregunta = None
                self.esperando_nombre = False
                
                return {
                    "estado": "esperando_marcador",
                    "mensajes": [
                        f"Elemento {progreso}/{total}",
                        f"Coloca el marcador ID: {self.marcador_actual}",
                        f"Busca: {info['nombre'] if info else 'Marcador'}"
                    ],
                    "esperando_marcadores": [self.marcador_actual],
                    "modelos": []
                }
        
        # FASE 3: JUEGO TERMINADO
        elif self.juego_terminado:
            return self._mostrar_resultado_final()
        
        return {"estado": "error", "mensajes": ["Estado no reconocido"]}
        
    def _timeout_respuesta(self):
        """Manejar cuando se agota el tiempo de respuesta"""
        info = obtener_info_modelo(self.marcador_actual)
        mensaje_resultado = f"Tiempo terminado. Era: {info['nombre']}"
        
        # Remover marcador actual
        self.marcadores_pendientes.remove(self.marcador_actual)
        self._reset_pregunta()
        
        # Preparar siguiente marcador
        if self.marcadores_pendientes:
            siguiente = self._mostrar_siguiente_marcador()
            siguiente["mensajes"].insert(0, mensaje_resultado)
            return siguiente
        else:
            # Si no hay más marcadores, finalizar juego
            return self._finalizar_juego_con_mensaje(mensaje_resultado)
    
    def _finalizar_juego(self):
        """Finaliza el juego y prepara el resultado final"""
        self.juego_terminado = True
        
        # Calcular estadísticas
        total_marcadores = len(self.marcadores_detectados_inicial)
        precision = (self.puntuacion/self.intentos)*100 if self.intentos > 0 else 0
        
        self.resultado_final = {
            "puntuacion": self.puntuacion,
            "total": total_marcadores,
            "intentos": self.intentos,
            "precision": precision
        }
        
        return self._mostrar_resultado_final()
    
    def _finalizar_juego_con_mensaje(self, mensaje_previo):
        """Finaliza el juego con un mensaje previo"""
        resultado = self._finalizar_juego()
        resultado["mensajes"].insert(0, mensaje_previo)
        return resultado
    
    def _mostrar_resultado_final(self):
        """Muestra el resultado final del juego"""
        if not self.resultado_final:
            return {"estado": "error", "mensajes": ["Error: No hay resultado"]}
        
        mensajes = [
            " ¡JUEGO COMPLETADO!",
            f"Puntuacion: {self.resultado_final['puntuacion']}/{self.resultado_final['total']}",
            f"Precision: {self.resultado_final['precision']:.1f}%",
            "¡Excelente trabajo!"
        ]
        
        # Mensaje adicional basado en la puntuación
        if self.resultado_final['precision'] >= 80:
            mensajes.append("¡Eres un experto en frutas y verduras!")
        elif self.resultado_final['precision'] >= 60:
            mensajes.append("¡Buen trabajo! Sigues mejorando")
        else:
            mensajes.append("¡Sigue practicando para mejorar!")
        
        mensajes.extend([
            "",
            "Di 'otra vez' para jugar de nuevo",
            "Di 'salir' para terminar"
        ])
        
        return {
            "estado": "resultado_final",
            "mensajes": mensajes,
            "esperando_marcadores": [],
            "modelos": []
        }
    
    def _reset_pregunta(self):
        """Resetear estado de pregunta"""
        self.pregunta_actual = None
        self.respuesta_correcta = None
        self.tiempo_pregunta = None
        self.esperando_nombre = False
        
    def _mostrar_siguiente_marcador(self):
        """Prepara el siguiente marcador del juego"""
        if not self.marcadores_pendientes:
            return self._finalizar_juego()
            
        self.marcador_actual = self.marcadores_pendientes[0] 
        self._reset_pregunta() 
        
        info = obtener_info_modelo(self.marcador_actual)
        progreso = len(self.marcadores_detectados_inicial) - len(self.marcadores_pendientes) + 1
        total = len(self.marcadores_detectados_inicial)
        
        return {
            "estado": "esperando_marcador",
            "mensajes": [
                f"Elemento {progreso}/{total}",
                f"Coloca el marcador ID: {self.marcador_actual}",
                "Cuando aparezca la fruta/verdura, di su nombre"
            ],
            "esperando_marcadores": [self.marcador_actual],
            "modelos": []
        }
        
    def procesar_comando(self, comando):
        comando_lower = comando.lower().strip()
        
        # Comandos especiales en resultado final
        if self.juego_terminado:
            if any(palabra in comando_lower for palabra in ['otra vez', 'repetir', 'nuevo', 'again']):
                # Reiniciar juego
                return self._inicializar_juego()
            elif any(palabra in comando_lower for palabra in ['salir', 'terminar', 'exit', 'quit']):
                return {
                    "estado": "salir",
                    "mensajes": [" ¡Gracias por jugar!", "Hasta la proxima"]
                }
            else:
                return None  # Ignorar otros comandos
        
        # Procesar respuesta durante el juego
        if not self.esperando_nombre or self.marcador_actual is None or not self.respuesta_correcta:
            return None
            
        info = obtener_info_modelo(self.marcador_actual)
        
        es_correcta = self.verificar_respuesta(comando, self.respuesta_correcta)
        
        self.intentos += 1
        
        if es_correcta:
            self.puntuacion += 1
            mensaje_resultado = f"¡CORRECTO! Es {info['nombre']}"
        else:
            mensaje_resultado = f"INCORRECTO era: {info['nombre']}. Dijiste: '{comando}'"
        
        # Remover marcador completado
        self.marcadores_pendientes.remove(self.marcador_actual)
        self._reset_pregunta()
        
        # Preparar siguiente marcador
        if self.marcadores_pendientes:
            siguiente = self._mostrar_siguiente_marcador()
            siguiente["mensajes"].insert(0, mensaje_resultado)
            siguiente["mensajes"].insert(1, "¡Muy bien! Preparando siguiente...")
            return siguiente
        else:
            # Finalizar juego
            return self._finalizar_juego_con_mensaje(mensaje_resultado)
        
    def obtener_estado_detallado(self):
        """Información detallada del juego"""
        base_info = super().obtener_estado_detallado()
        base_info.update({
            "marcadores_detectados": len(self.marcadores_detectados_inicial),
            "marcadores_pendientes": len(self.marcadores_pendientes),
            "marcador_actual": self.marcador_actual,
            "esperando_nombre": self.esperando_nombre,
            "fase_escaneo_completada": self.fase_escaneo_completada,
            "pregunta_actual": self.pregunta_actual,
            "respuesta_correcta": self.respuesta_correcta,
            "tiempo_restante": self.timeout_respuesta - (time.time() - self.tiempo_pregunta) if self.tiempo_pregunta else 0,
            "juego_terminado": self.juego_terminado
        })
        return base_info 

    def obtener_marcadores_renderizado(self):
        """
        Retorna qué marcadores se deben renderizar en 3D según el estado del juego
        """
        if not self.fase_escaneo_completada:
            # Durante escaneo: mostrar todos los marcadores válidos detectados
            return list(self.marcadores_detectados_inicial)
        elif not self.juego_terminado:
            # Durante juego: solo mostrar el marcador actual
            if self.marcador_actual:
                return [self.marcador_actual]
            return []
        else:
            # Durante resultado final: no mostrar marcadores
            return []

    def debe_escuchar_voz(self):
        """Determina si debe escuchar comandos de voz"""
        return (self.esperando_nombre and self.marcador_actual is not None and self.fase_escaneo_completada and not self.juego_terminado) or self.juego_terminado
     
class JuegoEncuentraFrutasAR(JuegoBaseAR):
    """Juego Encuentra las Frutas para AR"""
    
    def __init__(self):
        super().__init__()
        # Variables de control del escaneo inicial
        self.marcadores_detectados_inicial = set()
        self.tiempo_escaneo = None
        self.fase_escaneo_completada = False
        
        # Variables del juego
        self.frutas_objetivo = []
        self.marcadores_encontrados = set()
        self.esperando_nombres = False
        self.tiempo_pregunta = None
        # 30 segundos para responder
        self.timeout_respuesta = 30  
        self.nombres_dichos = []
        
        # Control de fin de juego
        self.juego_terminado = False
        self.resultado_final = None
        
    def obtener_nombre(self):
        return "ENCUENTRA LAS FRUTAS"
        
    def _inicializar_juego(self):
        """Inicia el juego con escaneo inicial"""
        self.marcadores_detectados_inicial.clear()
        self.tiempo_escaneo = time.time()
        self.fase_escaneo_completada = False
        self.frutas_objetivo = []
        self.marcadores_encontrados.clear()
        self.esperando_nombres = False
        self.tiempo_pregunta = None
        self.nombres_dichos = []
        self.juego_terminado = False
        self.resultado_final = None
        
        return {
            "estado": "escaneo_inicial",
            "mensajes": [
                "ESCANEO INICIAL (10 segundos)",
                "Muestra al menos 3 marcadores de frutas diferentes",
                "Colocalos uno por uno frente a la camara"
            ],
            "esperando_marcadores": [id for id, info in MODELOS_FRUTAS_VERDURAS.items() if info['tipo'] == 'fruta'],
        }
        
    def actualizar_marcadores(self, marcadores_visibles):
        """Actualiza según los marcadores detectados"""
        current_time = time.time()
        
        # FASE 1: ESCANEO INICIAL (primeros 10 segundos)
        if not self.fase_escaneo_completada:
            tiempo_transcurrido = current_time - self.tiempo_escaneo
            tiempo_restante = max(0, 10 - int(tiempo_transcurrido))
            
            # Añadir marcadores válidos de frutas detectados
            for marker_id in marcadores_visibles:
                if marker_id in MODELOS_FRUTAS_VERDURAS and MODELOS_FRUTAS_VERDURAS[marker_id]['tipo'] == 'fruta':
                    self.marcadores_detectados_inicial.add(marker_id)
            
            mensajes = [
                f"ESCANEO INICIAL - {tiempo_restante}s restantes",
                f"Frutas encontradas: {len(self.marcadores_detectados_inicial)}/3"
            ]
            
            # Mostrar qué frutas se han encontrado
            for marker_id in sorted(self.marcadores_detectados_inicial):
                info = obtener_info_modelo(marker_id)
                mensajes.append(f"{info['nombre']}")
            
            if tiempo_transcurrido >= 10:
                if len(self.marcadores_detectados_inicial) >= 3:
                    # Completar escaneo e iniciar juego
                    self.fase_escaneo_completada = True
                    # Seleccionar 3 frutas aleatorias
                    frutas_lista = list(self.marcadores_detectados_inicial)
                    self.frutas_objetivo = random.sample(frutas_lista, min(3, len(frutas_lista)))
                    
                    return self._iniciar_fase_juego()
                else:
                    mensajes.append(f"Necesitas al menos 3 frutas (tienes {len(self.marcadores_detectados_inicial)})")
                    # Reiniciar escaneo si no hay suficientes marcadores
                    if tiempo_transcurrido >= 15:
                        self.tiempo_escaneo = current_time
                        self.marcadores_detectados_inicial.clear()
            
            return {
                "estado": "escaneo_inicial",
                "mensajes": mensajes,
                "esperando_marcadores": [id for id, info in MODELOS_FRUTAS_VERDURAS.items() if info['tipo'] == 'fruta'],
                "modelos": [crear_modelo_por_id(mid) for mid in marcadores_visibles if mid in MODELOS_FRUTAS_VERDURAS and MODELOS_FRUTAS_VERDURAS[mid]['tipo'] == 'fruta']
            }
        
        # FASE 2: JUGANDO - Esperando que coloque las frutas objetivo
        elif not self.esperando_nombres:
            # Verificar qué frutas objetivo están presentes
            frutas_presentes = [m for m in marcadores_visibles if m in self.frutas_objetivo]
            
            if len(frutas_presentes) == len(self.frutas_objetivo):
                # Todas las frutas están presentes, pasar a fase de nombres
                self.esperando_nombres = True
                self.tiempo_pregunta = current_time
                
                nombres_frutas = [obtener_info_modelo(id)['nombre'] for id in self.frutas_objetivo]
                
                return {
                    "estado": "esperando_nombres",
                    "mensajes": [
                        "¡Perfecto! Todas las frutas estan aqui",
                        f"Ahora di los nombres: {', '.join(nombres_frutas)}",
                        "Puedes decirlos uno por uno o todos juntos"
                    ],
                    "esperando_marcadores": self.frutas_objetivo,
                    "modelos": [crear_modelo_por_id(m) for m in frutas_presentes]
                }
            else:
                nombres_objetivo = [obtener_info_modelo(id)['nombre'] for id in self.frutas_objetivo]
                return {
                    "estado": "colocando_frutas",
                    "mensajes": [
                        f"Coloca estas frutas: {', '.join(nombres_objetivo)}",
                        f"Tienes {len(frutas_presentes)}/{len(self.frutas_objetivo)} frutas"
                    ],
                    "esperando_marcadores": self.frutas_objetivo,
                    "modelos": [crear_modelo_por_id(m) for m in frutas_presentes]
                }
        
        # FASE 3: ESPERANDO NOMBRES
        elif self.esperando_nombres and not self.juego_terminado:
            return self._manejar_fase_nombres(current_time, marcadores_visibles)
        
        # FASE 4: JUEGO TERMINADO
        elif self.juego_terminado:
            return self._mostrar_resultado_final()
        
        return {"estado": "error", "mensajes": ["Estado no reconocido"]}
    
    def _iniciar_fase_juego(self):
        """Inicia la fase principal del juego"""
        nombres_frutas = [obtener_info_modelo(id)['nombre'] for id in self.frutas_objetivo]
        
        return {
            "estado": "colocando_frutas",
            "mensajes": [
                "¡ENCUENTRA ESTAS FRUTAS!",
                f"Busca y coloca: {', '.join(nombres_frutas)}",
                "Cuando las veas, di sus nombres en voz alta"
            ],
            "esperando_marcadores": self.frutas_objetivo,
            "modelos": []
        }
    
    def _manejar_fase_nombres(self, current_time, marcadores_visibles):
        """Maneja la fase donde el usuario debe decir los nombres"""
        if not self.esperando_nombres or self.juego_terminado:
            return self._mostrar_resultado_final()
        
        # Verificar que las frutas sigan presentes
        frutas_presentes = [m for m in marcadores_visibles if m in self.frutas_objetivo]
        
        if len(frutas_presentes) < len(self.frutas_objetivo):
            # Faltan frutas, volver a la fase anterior
            self.esperando_nombres = False
            nombres_objetivo = [obtener_info_modelo(id)['nombre'] for id in self.frutas_objetivo]
            return {
                "estado": "colocando_frutas",
                "mensajes": [
                    "¡Manten todas las frutas visibles!",
                    f"Coloca: {', '.join(nombres_objetivo)}"
                ],
                "esperando_marcadores": self.frutas_objetivo,
                "modelos": [crear_modelo_por_id(m) for m in frutas_presentes]
            }
        
        # Verificar timeout
        tiempo_esperando = current_time - self.tiempo_pregunta
        tiempo_restante = max(0, self.timeout_respuesta - int(tiempo_esperando))
        
        if tiempo_esperando >= self.timeout_respuesta:
            return self._timeout_respuesta()
        
        # Mostrar estado actual
        nombres_objetivo = [obtener_info_modelo(id)['nombre'] for id in self.frutas_objetivo]
        mensajes = [
            "DI LOS NOMBRES DE LAS FRUTAS",
            f"Frutas objetivo: {', '.join(nombres_objetivo)}",
            f"Tiempo restante: {tiempo_restante}s"
        ]
        
        if self.nombres_dichos:
            mensajes.append(f"Has dicho: {', '.join(self.nombres_dichos)}")
        
        return {
            "estado": "escuchando_nombres",
            "mensajes": mensajes,
            "esperando_marcadores": self.frutas_objetivo,
            "modelos": [crear_modelo_por_id(m) for m in frutas_presentes]
        }
    
    def _timeout_respuesta(self):
        """Maneja el timeout de respuesta"""
        nombres_correctos = [obtener_info_modelo(mid)['nombre'] for mid in self.frutas_objetivo]
        
        self.juego_terminado = True
        self.resultado_final = {
            "correcto": False,
            "razon": "timeout",
            "frutas_objetivo": nombres_correctos,
            "nombres_dichos": self.nombres_dichos
        }
        
        return self._mostrar_resultado_final()
    
    def _mostrar_resultado_final(self):
        """Muestra el resultado final del juego"""
        if not self.resultado_final:
            return {"estado": "error", "mensajes": ["Error: No hay resultado"]}
        
        mensajes = []
        
        if self.resultado_final["correcto"]:
            mensajes.extend([
                "¡EXCELENTE!",
                "¡Has encontrado todas las frutas!",
                f"Puntuación: {self.puntuacion}/{len(self.frutas_objetivo)}"
            ])
        else:
            if self.resultado_final["razon"] == "timeout":
                mensajes.append("Se acabó el tiempo")
            
            mensajes.extend([
                "Frutas objetivo:",
                ", ".join(self.resultado_final["frutas_objetivo"])
            ])
            
            if self.resultado_final["nombres_dichos"]:
                mensajes.extend([
                    "Dijiste:",
                    ", ".join(self.resultado_final["nombres_dichos"])
                ])
        
        mensajes.extend([
            "",
            "Di 'otra vez' para jugar de nuevo",
            "Di 'salir' para terminar"
        ])
        
        return {
            "estado": "resultado_final",
            "mensajes": mensajes,
            "esperando_marcadores": [],
            "modelos": []
        }
    
    def verificar_respuesta_elemento(self, texto, respuesta_correcta):
        """Verificar si un elemento de la respuesta es correcto"""
        texto = texto.lower().strip()
        respuesta_correcta = respuesta_correcta.lower().strip()
        
        # Verificaciones directas
        if respuesta_correcta in texto:
            return True
        
        # Verificaciones con variaciones y sinónimos
        variaciones = {
            'pera': ['pera'],
            'limon': ['limon', 'limón'],  
            'uvas': ['uvas', 'uva', 'racimo'],
            'manzana': ['manzana'],
            'naranja': ['naranja'],
            'platano': ['platano', 'plátano', 'banana'],
            'fresa': ['fresa', 'frutilla'],
            'sandia': ['sandia', 'sandía'],
            'melon': ['melon', 'melón'],
        }
        
        # Buscar variaciones
        if respuesta_correcta in variaciones:
            for variacion in variaciones[respuesta_correcta]:
                if variacion in texto:
                    return True
        
        # Verificación por palabras clave
        palabras_respuesta = respuesta_correcta.split()
        for palabra in palabras_respuesta:
            if len(palabra) > 2 and palabra in texto:
                return True
        
        return False
    
    def procesar_comando(self, comando):
        """Procesa comandos de voz durante el juego"""
        comando_lower = comando.lower().strip()
        
        # Comandos especiales en resultado final
        if self.juego_terminado:
            if any(palabra in comando_lower for palabra in ['otra vez', 'repetir', 'nuevo', 'again']):
                return self._inicializar_juego()
            elif any(palabra in comando_lower for palabra in ['salir', 'terminar', 'exit', 'quit']):
                return {
                    "estado": "salir",
                    "mensajes": [" ¡Gracias por jugar!", "Hasta la próxima"]
                }
            return None
        
        # Procesar nombres durante la fase de escucha
        if self.esperando_nombres and not self.juego_terminado:
            nombres_objetivo = [obtener_info_modelo(id)['nombre'].lower() for id in self.frutas_objetivo]
            
            # Buscar frutas mencionadas
            frutas_mencionadas = []
            for nombre in nombres_objetivo:
                nombre_original = obtener_info_modelo(
                    next(id for id in self.frutas_objetivo if obtener_info_modelo(id)['nombre'].lower() == nombre)
                )['nombre']
                
                if self.verificar_respuesta_elemento(comando, nombre_original) and nombre_original not in self.nombres_dichos:
                    frutas_mencionadas.append(nombre_original)
            
            # Agregar nuevas frutas mencionadas
            self.nombres_dichos.extend(frutas_mencionadas)
            
            # Verificar si ya se completó
            if len(self.nombres_dichos) >= len(self.frutas_objetivo):
                return self._evaluar_respuesta_completa()
        
        return None
    
    def _evaluar_respuesta_completa(self):
        """Evalúa la respuesta completa del usuario"""
        nombres_correctos = [obtener_info_modelo(mid)['nombre'].lower() for mid in self.frutas_objetivo]
        nombres_usuario = [n.lower() for n in self.nombres_dichos]
        
        # Contar aciertos
        aciertos = 0
        for nombre_correcto in nombres_correctos:
            if nombre_correcto in nombres_usuario:
                aciertos += 1
        
        # Actualizar puntuación
        self.puntuacion = aciertos
        self.intentos = len(self.frutas_objetivo)
        self.juego_terminado = True
        self.esperando_nombres = False
        
        self.resultado_final = {
            "correcto": aciertos == len(self.frutas_objetivo),
            "razon": "respuesta_completa",
            "frutas_objetivo": [obtener_info_modelo(mid)['nombre'] for mid in self.frutas_objetivo],
            "nombres_dichos": self.nombres_dichos
        }
        
        return self._mostrar_resultado_final()
    
    def obtener_estado_detallado(self):
        """Información detallada del juego"""
        base_info = super().obtener_estado_detallado()
        base_info.update({
            "marcadores_detectados": len(self.marcadores_detectados_inicial),
            "fase_escaneo_completada": self.fase_escaneo_completada,
            "frutas_objetivo": self.frutas_objetivo,
            "esperando_nombres": self.esperando_nombres,
            "nombres_dichos": self.nombres_dichos,
            "juego_terminado": self.juego_terminado
        })
        return base_info
    
    def obtener_marcadores_renderizado(self):
        """Retorna qué marcadores se deben renderizar en 3D según el estado del juego"""
        if not self.fase_escaneo_completada:
            # Durante escaneo: mostrar todas las frutas detectadas
            return list(self.marcadores_detectados_inicial)
        elif self.esperando_nombres or not self.juego_terminado:
            # Durante juego: mostrar las frutas objetivo
            return self.frutas_objetivo
        else:
            # Durante resultado: no mostrar marcadores
            return []
    
    def debe_escuchar_voz(self):
        """Determina si debe escuchar comandos de voz"""
        return (self.esperando_nombres and not self.juego_terminado) or self.juego_terminado

class JuegoCategoriasAR(JuegoBaseAR):
    """Juego de Categorías AR"""
    
    def __init__(self):
        super().__init__()
        # Variables de control del escaneo inicial
        self.marcadores_detectados_inicial = set()
        self.tiempo_escaneo = None
        self.fase_escaneo_completada = False
        
        # Variables del juego
        self.elementos_juego = []
        self.frutas_correctas = []
        self.verduras_correctas = []
        
        # Variables para recolección de respuestas
        # Empezamos con frutas
        self.categoria_actual = "frutas"  
        self.respuestas_frutas = []
        self.respuestas_verduras = []
        self.esperando_respuesta = False
        self.tiempo_pregunta = None
        # 30 segundos por categoría
        self.timeout_respuesta = 30  
        
        # Control de fin de juego
        self.juego_terminado = False
        self.resultado_final = None
        
    def obtener_nombre(self):
        return "AGRUPA POR CATEGORIAS"
        
    def _inicializar_juego(self):
        """Inicia el juego con escaneo inicial"""
        self.marcadores_detectados_inicial.clear()
        self.tiempo_escaneo = time.time()
        self.fase_escaneo_completada = False
        self.elementos_juego = []
        self.frutas_correctas = []
        self.verduras_correctas = []
        
        # Reiniciar variables de respuestas
        self.categoria_actual = "frutas"
        self.respuestas_frutas = []
        self.respuestas_verduras = []
        self.esperando_respuesta = False
        self.tiempo_pregunta = None
        
        self.juego_terminado = False
        self.resultado_final = None
        
        return {
            "estado": "escaneo_inicial",
            "mensajes": [
                "ESCANEO INICIAL (12 segundos)",
                "Muestra al menos 6 marcadores diferentes",
                "Mezcla de frutas y verduras"
            ],
            "esperando_marcadores": list(MODELOS_FRUTAS_VERDURAS.keys()),
        }
        
    def actualizar_marcadores(self, marcadores_visibles):
        """Actualiza segun los marcadores detectados"""
        current_time = time.time()
        
        # FASE 1: ESCANEO INICIAL (primeros 12 segundos)
        if not self.fase_escaneo_completada:
            tiempo_transcurrido = current_time - self.tiempo_escaneo
            tiempo_restante = max(0, 12 - int(tiempo_transcurrido))
            
            # Anadir marcadores validos detectados
            for marker_id in marcadores_visibles:
                if marker_id in MODELOS_FRUTAS_VERDURAS:
                    self.marcadores_detectados_inicial.add(marker_id)
            
            # Contar frutas y verduras
            frutas_count = sum(1 for mid in self.marcadores_detectados_inicial 
                             if MODELOS_FRUTAS_VERDURAS[mid]['tipo'] == 'fruta')
            verduras_count = sum(1 for mid in self.marcadores_detectados_inicial 
                               if MODELOS_FRUTAS_VERDURAS[mid]['tipo'] == 'verdura')
            
            mensajes = [
                f"ESCANEO INICIAL - {tiempo_restante}s restantes",
                f"Elementos encontrados: {len(self.marcadores_detectados_inicial)}/6",
                f"Frutas: {frutas_count}, Verduras: {verduras_count}"
            ]
            
            # Mostrar elementos encontrados
            for marker_id in sorted(self.marcadores_detectados_inicial):
                info = obtener_info_modelo(marker_id)
                tipo_icono = "[FRUTA]" if info['tipo'] == 'fruta' else "[VERDURA]"
                mensajes.append(f"{tipo_icono} {info['nombre']}")
            
            if tiempo_transcurrido >= 12:
                if len(self.marcadores_detectados_inicial) >= 6:
                    # Completar escaneo e iniciar juego
                    self.fase_escaneo_completada = True
                    return self._iniciar_fase_juego()
                else:
                    mensajes.append(f"Necesitas al menos 6 elementos (tienes {len(self.marcadores_detectados_inicial)})")
                    # Reiniciar escaneo si no hay suficientes marcadores
                    if tiempo_transcurrido >= 17:
                        self.tiempo_escaneo = current_time
                        self.marcadores_detectados_inicial.clear()
            
            return {
                "estado": "escaneo_inicial",
                "mensajes": mensajes,
                "esperando_marcadores": list(MODELOS_FRUTAS_VERDURAS.keys()),
                "modelos": [crear_modelo_por_id(mid) for mid in marcadores_visibles if mid in MODELOS_FRUTAS_VERDURAS]
            }
        
        # FASE 2: JUGANDO - Esperando que coloque todos los elementos
        elif not self.esperando_respuesta and not self.juego_terminado:
            elementos_presentes = [m for m in marcadores_visibles if m in self.elementos_juego]
            
            if len(elementos_presentes) == len(self.elementos_juego):
                # Todos los elementos estan presentes, comenzar preguntas
                return self._iniciar_primera_categoria()
            else:
                return {
                    "estado": "colocando_elementos",
                    "mensajes": [
                        "Coloca todos los elementos escaneados",
                        f"Tienes {len(elementos_presentes)}/{len(self.elementos_juego)} elementos"
                    ],
                    "esperando_marcadores": self.elementos_juego,
                    "modelos": [crear_modelo_por_id(m) for m in elementos_presentes]
                }
        
        # FASE 3: ESPERANDO RESPUESTA POR CATEGORIA
        elif self.esperando_respuesta and not self.juego_terminado:
            return self._manejar_categoria_actual(current_time, marcadores_visibles)
        
        # FASE 4: JUEGO TERMINADO
        elif self.juego_terminado:
            return self._mostrar_resultado_final()
        
        return {"estado": "error", "mensajes": ["Estado no reconocido"]}
    
    def _iniciar_fase_juego(self):
        """Inicia la fase principal del juego"""
        # Seleccionar elementos del escaneo
        elementos_lista = list(self.marcadores_detectados_inicial)
        
        # Asegurar balance entre frutas y verduras
        frutas_disponibles = [id for id in elementos_lista if MODELOS_FRUTAS_VERDURAS[id]['tipo'] == 'fruta']
        verduras_disponibles = [id for id in elementos_lista if MODELOS_FRUTAS_VERDURAS[id]['tipo'] == 'verdura']
        
        # Seleccionar 3 frutas y 3 verduras si es posible
        frutas_sel = frutas_disponibles[:3] if len(frutas_disponibles) >= 3 else frutas_disponibles
        verduras_sel = verduras_disponibles[:3] if len(verduras_disponibles) >= 3 else verduras_disponibles
        
        # Completar con elementos adicionales si es necesario
        elementos_seleccionados = frutas_sel + verduras_sel
        if len(elementos_seleccionados) < 6:
            elementos_faltantes = [id for id in elementos_lista if id not in elementos_seleccionados]
            elementos_seleccionados.extend(elementos_faltantes[:6-len(elementos_seleccionados)])
        
        self.elementos_juego = elementos_seleccionados
        random.shuffle(self.elementos_juego)
        
        # Preparar listas de respuestas correctas
        self.frutas_correctas = [obtener_info_modelo(id)['nombre'] for id in self.elementos_juego 
                               if MODELOS_FRUTAS_VERDURAS[id]['tipo'] == 'fruta']
        self.verduras_correctas = [obtener_info_modelo(id)['nombre'] for id in self.elementos_juego 
                                 if MODELOS_FRUTAS_VERDURAS[id]['tipo'] == 'verdura']
        
        return {
            "estado": "colocando_elementos",
            "mensajes": [
                "CLASIFICA EN FRUTAS Y VERDURAS!",
                f"Coloca estos {len(self.elementos_juego)} elementos",
                "Te preguntare por cada categoria"
            ],
            "esperando_marcadores": self.elementos_juego,
            "modelos": []
        }
    
    def _iniciar_primera_categoria(self):
        """Inicia la primera categoria de preguntas"""
        self.categoria_actual = "frutas"
        self.esperando_respuesta = True
        self.tiempo_pregunta = time.time()
        
        return self._mostrar_categoria_actual()
    
    def _mostrar_categoria_actual(self):
        """Muestra la pregunta de la categoria actual"""
        # Calcular tiempo restante
        tiempo_transcurrido = time.time() - self.tiempo_pregunta
        tiempo_restante = max(0, self.timeout_respuesta - int(tiempo_transcurrido))
        
        if self.categoria_actual == "frutas":
            icono = "[FRUTA]"
            categoria_nombre = "FRUTAS"
            total_categoria = len(self.frutas_correctas)
            respuestas_actuales = len(self.respuestas_frutas)
        else:
            icono = "[VERDURA]"
            categoria_nombre = "VERDURAS"
            total_categoria = len(self.verduras_correctas)
            respuestas_actuales = len(self.respuestas_verduras)
        
        mensajes = [
            f"{icono} DIME TODAS LAS {categoria_nombre}",
            f"Di una por una (en cualquier orden)",
            f"Tiempo: {tiempo_restante}s",
            f"Encontradas: {respuestas_actuales}/{total_categoria}"
        ]
        
        # Mostrar las que ya ha dicho
        if self.categoria_actual == "frutas" and self.respuestas_frutas:
            mensajes.append("Ya dijiste: " + ", ".join(self.respuestas_frutas))
        elif self.categoria_actual == "verduras" and self.respuestas_verduras:
            mensajes.append("Ya dijiste: " + ", ".join(self.respuestas_verduras))
        
        mensajes.append("Di 'siguiente' para pasar a verduras")
        mensajes.append("Di 'listo' cuando termines")
        
        return {
            "estado": f"categoria_{self.categoria_actual}",
            "mensajes": mensajes,
            "esperando_marcadores": self.elementos_juego,
            "modelos": [crear_modelo_por_id(m) for m in self.elementos_juego]
        }
    
    def _manejar_categoria_actual(self, current_time, marcadores_visibles):
        """Maneja la categoría actual"""
        # Verificar que todos los elementos sigan presentes
        elementos_presentes = [m for m in marcadores_visibles if m in self.elementos_juego]
        
        if len(elementos_presentes) < len(self.elementos_juego):
            # Faltan elementos, pausar pregunta
            self.esperando_respuesta = False
            return {
                "estado": "colocando_elementos",
                "mensajes": [
                    "¡Manten todos los elementos visibles!",
                    f"Faltan {len(self.elementos_juego) - len(elementos_presentes)} elementos",
                    "Pon todos para continuar con las preguntas"
                ],
                "esperando_marcadores": self.elementos_juego,
                "modelos": [crear_modelo_por_id(m) for m in elementos_presentes]
            }
        
        # Verificar timeout
        tiempo_esperando = current_time - self.tiempo_pregunta
        
        if tiempo_esperando >= self.timeout_respuesta:
            return self._timeout_categoria_actual()
        
        # Mostrar categoría actual
        return self._mostrar_categoria_actual()
    
    def _timeout_categoria_actual(self):
        """Maneja el timeout de la categoría actual"""
        if self.categoria_actual == "frutas":
            # Pasar a verduras
            return self._cambiar_a_verduras()
        else:
            # Finalizar juego
            return self._finalizar_juego()
    
    def _cambiar_a_verduras(self):
        """Cambia a la categoría de verduras"""
        self.categoria_actual = "verduras"
        self.tiempo_pregunta = time.time()
        return self._mostrar_categoria_actual()
    
    def _finalizar_juego(self):
        """Finaliza el juego y calcula resultados"""
        self.juego_terminado = True
        self.esperando_respuesta = False
        
        # Verificar respuestas correctas
        frutas_correctas_encontradas = []
        frutas_incorrectas = []
        
        for respuesta in self.respuestas_frutas:
            encontrada = False
            for fruta_correcta in self.frutas_correctas:
                if self.verificar_elemento_categoria(respuesta, fruta_correcta):
                    if fruta_correcta not in frutas_correctas_encontradas:
                        frutas_correctas_encontradas.append(fruta_correcta)
                        encontrada = True
                        break
            if not encontrada:
                frutas_incorrectas.append(respuesta)
        
        verduras_correctas_encontradas = []
        verduras_incorrectas = []
        
        for respuesta in self.respuestas_verduras:
            encontrada = False
            for verdura_correcta in self.verduras_correctas:
                if self.verificar_elemento_categoria(respuesta, verdura_correcta):
                    if verdura_correcta not in verduras_correctas_encontradas:
                        verduras_correctas_encontradas.append(verdura_correcta)
                        encontrada = True
                        break
            if not encontrada:
                verduras_incorrectas.append(respuesta)
        
        # Calcular puntuación
        total_correctas = len(frutas_correctas_encontradas) + len(verduras_correctas_encontradas)
        total_elementos = len(self.frutas_correctas) + len(self.verduras_correctas)
        
        self.puntuacion = total_correctas
        self.intentos += 1
        
        self.resultado_final = {
            "correcto": total_correctas >= (total_elementos * 0.8), 
            "total_correctas": total_correctas,
            "total_elementos": total_elementos,
            "porcentaje": (total_correctas / total_elementos * 100) if total_elementos > 0 else 0,
            "frutas_correctas_encontradas": frutas_correctas_encontradas,
            "frutas_incorrectas": frutas_incorrectas,
            "frutas_perdidas": [f for f in self.frutas_correctas if f not in frutas_correctas_encontradas],
            "verduras_correctas_encontradas": verduras_correctas_encontradas,
            "verduras_incorrectas": verduras_incorrectas,
            "verduras_perdidas": [v for v in self.verduras_correctas if v not in verduras_correctas_encontradas],
            "respuestas_frutas": self.respuestas_frutas,
            "respuestas_verduras": self.respuestas_verduras
        }
        
        return self._mostrar_resultado_final()
    
    def _mostrar_resultado_final(self):
        """Muestra el resultado final del juego"""
        if not self.resultado_final:
            return {"estado": "error", "mensajes": ["Error: No hay resultado"]}
        
        resultado = self.resultado_final
        mensajes = []
        
        if resultado["correcto"]:
            mensajes.extend([
                "¡EXCELENTE CLASIFICACION!",
                f"Puntuacion: {resultado['total_correctas']}/{resultado['total_elementos']}"
            ])
        else:
            mensajes.extend([
                "¡Buen intento!",
                f"Puntuacion: {resultado['total_correctas']}/{resultado['total_elementos']}",
                f"Porcentaje: {resultado['porcentaje']:.1f}%"
            ])
        
        # Mostrar detalles de frutas
        mensajes.append("")
        mensajes.append(" FRUTAS:")
        if resultado['frutas_correctas_encontradas']:
            mensajes.append(f" Correctas: {', '.join(resultado['frutas_correctas_encontradas'])}")
        if resultado['frutas_perdidas']:
            mensajes.append(f" No encontradas: {', '.join(resultado['frutas_perdidas'])}")
        if resultado['frutas_incorrectas']:
            mensajes.append(f" Incorrectas: {', '.join(resultado['frutas_incorrectas'])}")
        
        # Mostrar detalles de verduras
        mensajes.append("")
        mensajes.append(" VERDURAS:")
        if resultado['verduras_correctas_encontradas']:
            mensajes.append(f" Correctas: {', '.join(resultado['verduras_correctas_encontradas'])}")
        if resultado['verduras_perdidas']:
            mensajes.append(f" No encontradas: {', '.join(resultado['verduras_perdidas'])}")
        if resultado['verduras_incorrectas']:
            mensajes.append(f" Incorrectas: {', '.join(resultado['verduras_incorrectas'])}")
        
        # Mostrar todas las respuestas correctas
        mensajes.append("")
        mensajes.append("RESPUESTAS CORRECTAS COMPLETAS:")
        mensajes.append(f"Frutas: {', '.join(self.frutas_correctas)}")
        mensajes.append(f"Verduras: {', '.join(self.verduras_correctas)}")
        
        mensajes.extend([
            "",
            "Di 'otra vez' para jugar de nuevo",
            "Di 'salir' para terminar"
        ])
        
        return {
            "estado": "resultado_final",
            "mensajes": mensajes,
            "esperando_marcadores": self.elementos_juego,
            "modelos": [crear_modelo_por_id(m) for m in self.elementos_juego]
        }
    
    def verificar_elemento_categoria(self, texto, elemento_nombre):
        """Verificar si un elemento se menciona en el texto"""
        texto = texto.lower().strip()
        elemento_nombre = elemento_nombre.lower().strip()
        
        # Verificaciones directas
        if elemento_nombre in texto:
            return True
        
        # Verificaciones con variaciones y sinónimos
        variaciones = {
            'pera': ['pera'],
            'cebolleta': ['cebolleta', 'cebollino', 'cebolla verde'],
            'cebolla': ['cebolla'],
            'lechuga': ['lechuga'],
            'limon': ['limon', 'limón'],  
            'pimiento rojo': ['pimiento rojo', 'pimiento', 'pimentón rojo', 'chile rojo'],
            'pimiento verde': ['pimiento verde', 'pimiento', 'pimentón verde', 'chile verde'],
            'uvas': ['uvas', 'uva', 'racimo'],
            'zanahoria': ['zanahoria'],
        }
        
        # Buscar variaciones
        if elemento_nombre in variaciones:
            for variacion in variaciones[elemento_nombre]:
                if variacion in texto:
                    return True
        
        # Verificación por palabras clave
        palabras_elemento = elemento_nombre.split()
        for palabra in palabras_elemento:
            if len(palabra) > 2 and palabra in texto:
                return True
        
        return False
    
    def procesar_comando(self, comando):
        """Procesa comandos de voz durante el juego"""
        comando_lower = comando.lower().strip()
        
        # Comandos especiales en resultado final
        if self.juego_terminado:
            if any(palabra in comando_lower for palabra in ['otra vez', 'repetir', 'nuevo', 'again']):
                # Reiniciar juego
                return self._inicializar_juego()
            elif any(palabra in comando_lower for palabra in ['salir', 'terminar', 'exit', 'quit']):
                return {
                    "estado": "salir",
                    "mensajes": ["¡Gracias por jugar!", "Hasta la proxima"]
                }
            else:
                return None 
        
        # Procesar respuesta durante categoría
        if self.esperando_respuesta and not self.juego_terminado:
            return self._procesar_respuesta_categoria(comando_lower)
        
        return None
    
    def _procesar_respuesta_categoria(self, comando):
        """Procesa la respuesta durante una categoría"""
        # Comandos especiales
        if any(palabra in comando for palabra in ['siguiente', 'verduras', 'cambiar']):
            if self.categoria_actual == "frutas":
                return self._cambiar_a_verduras()
            else:
                return self._finalizar_juego()
        
        if any(palabra in comando for palabra in ['listo', 'terminar', 'acabar', 'finalizar']):
            return self._finalizar_juego()
        
        # Procesar respuesta normal
        if self.categoria_actual == "frutas":
            # Verificar si ya se dijo esta respuesta
            if comando not in self.respuestas_frutas:
                self.respuestas_frutas.append(comando)
        else:
            # Verificar si ya se dijo esta respuesta
            if comando not in self.respuestas_verduras:
                self.respuestas_verduras.append(comando)
        
        # Continuar mostrando la categoría actual
        return self._mostrar_categoria_actual()
    
    def obtener_estado_detallado(self):
        """Información detallada del juego"""
        base_info = super().obtener_estado_detallado()
        base_info.update({
            "marcadores_detectados": len(self.marcadores_detectados_inicial),
            "fase_escaneo_completada": self.fase_escaneo_completada,
            "elementos_juego": self.elementos_juego,
            "categoria_actual": self.categoria_actual,
            "esperando_respuesta": self.esperando_respuesta,
            "respuestas_frutas": len(self.respuestas_frutas),
            "respuestas_verduras": len(self.respuestas_verduras),
            "frutas_correctas_total": len(self.frutas_correctas),
            "verduras_correctas_total": len(self.verduras_correctas),
            "juego_terminado": self.juego_terminado
        })
        return base_info
    
    def obtener_marcadores_renderizado(self):
        """Retorna qué marcadores se deben renderizar en 3D según el estado del juego"""
        if not self.fase_escaneo_completada:
            # Durante escaneo: mostrar todos los marcadores válidos detectados
            return list(self.marcadores_detectados_inicial)
        elif self.elementos_juego:
            # Durante juego: mostrar todos los elementos del juego
            return self.elementos_juego
        else:
            # Estado por defecto: no mostrar marcadores
            return []
    
    def debe_escuchar_voz(self):
        """Determina si debe escuchar comandos de voz"""
        return (self.esperando_respuesta and not self.juego_terminado) or self.juego_terminado



