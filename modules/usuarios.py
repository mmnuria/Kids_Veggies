import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

DB_PATH = "data/usuarios.json"
# Umbral para considerar que dos caras son la misma persona
UMBRAL_SIMILITUD = 0.94  

def cargar_usuarios():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_usuarios(data):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def obtener_usuario(nombre):
    usuarios = cargar_usuarios()
    return usuarios.get(nombre.lower())

def registrar_usuario(nombre, idioma, vector_facial=None):
    """
    Registra un usuario con idioma y opcionalmente con vector facial
    """
    usuarios = cargar_usuarios()
    nombre_key = nombre.lower()
    
    if nombre_key not in usuarios:
        nuevo_usuario = {
            "nombre": nombre,
            "idioma": idioma,
            "juegos": {
                "entrenamiento": {},
                "evaluacion": {}
            },
            "fecha_registro": datetime.now().isoformat()
        }
        
        # Agregar vector facial si se proporciona
        if vector_facial is not None:
            nuevo_usuario["vector_facial"] = vector_facial
        
        usuarios[nombre_key] = nuevo_usuario
        guardar_usuarios(usuarios)
        print(f"✅ Usuario {nombre} registrado correctamente")
    
    return usuarios[nombre_key]

def guardar_puntuacion_juego(nombre_usuario, modo, nombre_juego, puntuacion_obtenida):
    """
    Guarda la puntuación en porcentaje de un juego específico y actualiza las estadísticas
    
    Args:
        nombre_usuario (str): Nombre del usuario
        modo (str): 'entrenamiento' o 'evaluacion'
        nombre_juego (str): Nombre del juego
        puntuacion_obtenida (float): Porcentaje de aciertos en esta partida (0–100)
    """
    usuarios = cargar_usuarios()
    nombre_key = nombre_usuario.lower()
    
    if nombre_key not in usuarios:
        print(f"Usuario {nombre_usuario} no encontrado")
        return False
    
    usuario = usuarios[nombre_key]

    try:
        puntuacion_obtenida = float(puntuacion_obtenida)
    except (ValueError, TypeError):
        print(f"Puntuación inválida: {puntuacion_obtenida}")
        return False

    # Asegurar que existe la estructura de juegos
    if "juegos" not in usuario:
        usuario["juegos"] = {"entrenamiento": {}, "evaluacion": {}}
    
    if modo not in usuario["juegos"]:
        usuario["juegos"][modo] = {}

    # Inicializar si es la primera vez
    if nombre_juego not in usuario["juegos"][modo]:
        usuario["juegos"][modo][nombre_juego] = {
            "puntuacion_media": puntuacion_obtenida,
            "partidas_jugadas": 1,
            "mejor_puntuacion": puntuacion_obtenida,
            "ultima_puntuacion": puntuacion_obtenida,
            "suma_porcentajes": puntuacion_obtenida,
            "fecha_ultima_partida": datetime.now().isoformat(),
            "fecha_primera_partida": datetime.now().isoformat()
        }
    else:
        juego_stats = usuario["juegos"][modo][nombre_juego]

        try:
            juego_stats["suma_porcentajes"] = float(juego_stats.get("suma_porcentajes", 0.0))
            juego_stats["mejor_puntuacion"] = float(juego_stats.get("mejor_puntuacion", 0.0))
            juego_stats["partidas_jugadas"] = int(juego_stats.get("partidas_jugadas", 0))
        except (ValueError, TypeError) as e:
            print(f" Error al convertir estadisticas existentes: {e}")
            juego_stats["suma_porcentajes"] = puntuacion_obtenida
            juego_stats["partidas_jugadas"] = 1
            juego_stats["mejor_puntuacion"] = puntuacion_obtenida

        juego_stats["partidas_jugadas"] += 1
        juego_stats["suma_porcentajes"] += puntuacion_obtenida

        nueva_media = juego_stats["suma_porcentajes"] / juego_stats["partidas_jugadas"]

        juego_stats["puntuacion_media"] = round(nueva_media, 2)
        juego_stats["mejor_puntuacion"] = max(juego_stats["mejor_puntuacion"], puntuacion_obtenida)
        juego_stats["ultima_puntuacion"] = puntuacion_obtenida
        juego_stats["fecha_ultima_partida"] = datetime.now().isoformat()

    # Guardar cambios
    try:
        guardar_usuarios(usuarios)
    except Exception as e:
        print(f"Error al guardar usuarios: {e}")
        return False

    stats = usuario["juegos"][modo][nombre_juego]
    print(f"✅ Puntuación guardada para {nombre_usuario}")
    print(f"   Juego: {nombre_juego} ({modo})")
    print(f"   Ultima partida: {puntuacion_obtenida:.1f}%")
    print(f"   Media actual: {stats['puntuacion_media']:.2f}%")
    print(f"   Partidas jugadas: {stats['partidas_jugadas']}")
    print(f"   Mejor puntuación: {stats['mejor_puntuacion']:.1f}%")
    
    return True

def obtener_estadisticas_juego(nombre_usuario, modo=None, nombre_juego=None):
    """
    Obtiene estadísticas de juegos de un usuario
    
    Args:
        nombre_usuario (str): Nombre del usuario
        modo (str, optional): 'entrenamiento', 'evaluacion' o None para ambos
        nombre_juego (str, optional): Nombre específico del juego o None para todos
    
    Returns:
        dict: Estadísticas solicitadas
    """
    usuario = obtener_usuario(nombre_usuario)
    if not usuario:
        return None
    
    if "juegos" not in usuario:
        return {"mensaje": "El usuario no ha jugado ningún juego aún"}
    
    estadisticas = {}
    
    # Si se especifica modo
    if modo:
        if modo in usuario["juegos"]:
            if nombre_juego:
                # Juego específico en modo específico
                if nombre_juego in usuario["juegos"][modo]:
                    estadisticas[f"{modo}_{nombre_juego}"] = usuario["juegos"][modo][nombre_juego]
                else:
                    return {"mensaje": f"No se encontraron datos para {nombre_juego} en modo {modo}"}
            else:
                # Todos los juegos en modo específico
                estadisticas[modo] = usuario["juegos"][modo]
        else:
            return {"mensaje": f"No se encontraron datos para el modo {modo}"}
    else:
        # Todos los modos
        if nombre_juego:
            # Juego específico en todos los modos
            for modo_key in ["entrenamiento", "evaluacion"]:
                if modo_key in usuario["juegos"] and nombre_juego in usuario["juegos"][modo_key]:
                    estadisticas[f"{modo_key}_{nombre_juego}"] = usuario["juegos"][modo_key][nombre_juego]
        else:
            # Todos los juegos en todos los modos
            estadisticas = usuario["juegos"]
    
    return estadisticas

def obtener_ranking_juego(nombre_juego, modo, top=10):
    """
    Obtiene un ranking de usuarios para un juego específico
    
    Args:
        nombre_juego (str): Nombre del juego
        modo (str): 'entrenamiento' o 'evaluacion'
        top (int): Número de usuarios a mostrar en el ranking
    
    Returns:
        list: Lista de usuarios ordenados por puntuación media
    """
    usuarios = cargar_usuarios()
    ranking = []
    
    for nombre_key, usuario_data in usuarios.items():
        if ("juegos" in usuario_data and 
            modo in usuario_data["juegos"] and 
            nombre_juego in usuario_data["juegos"][modo]):
            
            stats = usuario_data["juegos"][modo][nombre_juego]
            ranking.append({
                "nombre": usuario_data.get("nombre", nombre_key),
                "puntuacion_media": stats["puntuacion_media"],
                "partidas_jugadas": stats["partidas_jugadas"],
                "mejor_puntuacion": stats["mejor_puntuacion"],
                "fecha_ultima_partida": stats["fecha_ultima_partida"]
            })
    
    # Ordenar por puntuación media (descendente)
    ranking.sort(key=lambda x: x["puntuacion_media"], reverse=True)
    
    return ranking[:top]

def obtener_progreso_usuario(nombre_usuario):
    """
    Obtiene un resumen completo del progreso de un usuario
    
    Args:
        nombre_usuario (str): Nombre del usuario
    
    Returns:
        dict: Resumen completo de progreso
    """
    usuario = obtener_usuario(nombre_usuario)
    if not usuario:
        return None
    
    if "juegos" not in usuario:
        return {"mensaje": "El usuario no ha jugado ningún juego aún"}
    
    progreso = {
        "nombre": usuario.get("nombre", nombre_usuario),
        "fecha_registro": usuario.get("fecha_registro", "No disponible"),
        "resumen_por_modo": {},
        "total_partidas": 0,
        "juegos_completados": 0
    }
    
    for modo in ["entrenamiento", "evaluacion"]:
        if modo in usuario["juegos"]:
            modo_stats = {
                "juegos_jugados": len(usuario["juegos"][modo]),
                "total_partidas_modo": 0,
                "promedio_general": 0,
                "mejor_juego": None,
                "juegos": {}
            }
            
            total_puntuacion_modo = 0
            total_partidas_modo = 0
            mejor_puntuacion_modo = 0
            
            for nombre_juego, stats in usuario["juegos"][modo].items():
                modo_stats["juegos"][nombre_juego] = stats
                total_partidas_modo += stats["partidas_jugadas"]
                total_puntuacion_modo += stats["puntuacion_media"] * stats["partidas_jugadas"]
                
                if stats["puntuacion_media"] > mejor_puntuacion_modo:
                    mejor_puntuacion_modo = stats["puntuacion_media"]
                    modo_stats["mejor_juego"] = {
                        "nombre": nombre_juego,
                        "puntuacion": stats["puntuacion_media"]
                    }
            
            if total_partidas_modo > 0:
                modo_stats["promedio_general"] = round(total_puntuacion_modo / total_partidas_modo, 2)
            
            modo_stats["total_partidas_modo"] = total_partidas_modo
            progreso["resumen_por_modo"][modo] = modo_stats
            progreso["total_partidas"] += total_partidas_modo
            progreso["juegos_completados"] += len(usuario["juegos"][modo])
    
    return progreso

def imprimir_progreso_usuario(nombre_usuario):
    """
    Imprime un resumen formateado del progreso del usuario
    """
    progreso = obtener_progreso_usuario(nombre_usuario)
    if not progreso:
        print(f"❌ Usuario {nombre_usuario} no encontrado")
        return
    
    if "mensaje" in progreso:
        print(f" {progreso['mensaje']}")
        return
    
    print(f"\n PROGRESO DE {progreso['nombre'].upper()}")
    print(f" Registrado: {progreso['fecha_registro'][:10] if progreso['fecha_registro'] != 'No disponible' else 'No disponible'}")
    print(f" Total partidas: {progreso['total_partidas']}")
    print(f" Juegos diferentes jugados: {progreso['juegos_completados']}")
    
    for modo, stats in progreso["resumen_por_modo"].items():
        print(f"\n MODO {modo.upper()}:")
        print(f"   Juegos jugados: {stats['juegos_jugados']}")
        print(f"   Partidas totales: {stats['total_partidas_modo']}")
        print(f"   Promedio general: {stats['promedio_general']:.1f}%")
        
        if stats["mejor_juego"]:
            print(f"   Mejor juego: {stats['mejor_juego']['nombre']} ({stats['mejor_juego']['puntuacion']:.1f}%)")
        
        print(f"   Detalle por juego:")
        for nombre_juego, juego_stats in stats["juegos"].items():
            print(f"- {nombre_juego}: {juego_stats['puntuacion_media']:.1f}% (promedio de {juego_stats['partidas_jugadas']} partidas)")

def comparar_vectores_faciales(vector1, vector2):
    """Compara dos vectores faciales y retorna la similitud"""
    try:
        if vector1 is None or vector2 is None:
            return 0.0
        
        # Convertir a arrays numpy
        v1 = np.array(vector1).reshape(1, -1)
        v2 = np.array(vector2).reshape(1, -1)
        
        # Calcular similitud coseno
        similitud = cosine_similarity(v1, v2)[0][0]
        return similitud
    except Exception as e:
        print(f"Error comparando vectores: {e}")
        return 0.0

def buscar_usuario_por_cara(vector_facial):
    """
    Busca un usuario por su vector facial
    Retorna el nombre del usuario y sus datos si encuentra coincidencia
    """
    if vector_facial is None:
        return None, None
        
    usuarios = cargar_usuarios()
    
    for nombre_key, datos in usuarios.items():
        if 'vector_facial' in datos and datos['vector_facial'] is not None:
            similitud = comparar_vectores_faciales(vector_facial, datos['vector_facial'])
            print(f"Similitud con {datos.get('nombre', nombre_key)}: {similitud:.3f}")
            
            if similitud >= UMBRAL_SIMILITUD:
                print(f"✅ Usuario {datos.get('nombre', nombre_key)} reconocido por cara")
                return datos.get('nombre', nombre_key), datos
    
    print("❌ No se encontró usuario con esa cara")
    return None, None

def actualizar_vector_facial(nombre, vector_facial):
    """
    Actualiza el vector facial de un usuario existente
    """
    usuarios = cargar_usuarios()
    nombre_key = nombre.lower()
    
    if nombre_key in usuarios:
        usuarios[nombre_key]["vector_facial"] = vector_facial
        usuarios[nombre_key]["fecha_actualizacion_facial"] = datetime.now().isoformat()
        guardar_usuarios(usuarios)
        print(f"✅ Vector facial actualizado para {nombre}")
        return True
    else:
        print(f"❌ Usuario {nombre} no encontrado")
        return False

def eliminar_vector_facial(nombre):
    """
    Elimina el vector facial de un usuario (mantiene el resto de datos)
    """
    usuarios = cargar_usuarios()
    nombre_key = nombre.lower()
    
    if nombre_key in usuarios:
        if "vector_facial" in usuarios[nombre_key]:
            del usuarios[nombre_key]["vector_facial"]
            usuarios[nombre_key]["fecha_eliminacion_facial"] = datetime.now().isoformat()
            guardar_usuarios(usuarios)
            print(f"✅ Vector facial eliminado para {nombre}")
            return True
        else:
            print(f"⚠️ Usuario {nombre} no tiene vector facial")
            return False
    else:
        print(f"❌ Usuario {nombre} no encontrado")
        return False

def listar_usuarios_con_cara():
    """
    Lista todos los usuarios que tienen vector facial registrado
    """
    usuarios = cargar_usuarios()
    usuarios_con_cara = []
    
    for nombre_key, datos in usuarios.items():
        if 'vector_facial' in datos and datos['vector_facial'] is not None:
            usuarios_con_cara.append({
                'nombre': datos.get('nombre', nombre_key),
                'idioma': datos.get('idioma', 'No especificado'),
                'fecha_registro': datos.get('fecha_registro', 'No disponible')
            })
    
    return usuarios_con_cara

def verificar_usuario_existe(nombre):
    """
    Verifica si un usuario ya existe en la base de datos
    """
    usuarios = cargar_usuarios()
    return nombre.lower() in usuarios

def configurar_umbral_similitud(nuevo_umbral):
    """
    Configura el umbral de similitud para el reconocimiento facial
    Valores recomendados: 0.4 (permisivo) a 0.8 (estricto)
    """
    global UMBRAL_SIMILITUD
    if 0.0 <= nuevo_umbral <= 1.0:
        UMBRAL_SIMILITUD = nuevo_umbral
        print(f"✅ Umbral de similitud configurado a {nuevo_umbral}")
        return True
    else:
        print("❌ El umbral debe estar entre 0.0 y 1.0")
        return False

def obtener_estadisticas_usuarios():
    """
    Obtiene estadísticas generales de los usuarios
    """
    usuarios = cargar_usuarios()
    
    total_usuarios = len(usuarios)
    usuarios_con_cara = sum(1 for datos in usuarios.values() 
                           if 'vector_facial' in datos and datos['vector_facial'] is not None)
    usuarios_sin_cara = total_usuarios - usuarios_con_cara
    
    idiomas = {}
    for datos in usuarios.values():
        idioma = datos.get('idioma', 'No especificado')
        idiomas[idioma] = idiomas.get(idioma, 0) + 1
    
    return {
        'total_usuarios': total_usuarios,
        'usuarios_con_reconocimiento_facial': usuarios_con_cara,
        'usuarios_sin_reconocimiento_facial': usuarios_sin_cara,
        'distribucion_idiomas': idiomas,
        'umbral_similitud_actual': UMBRAL_SIMILITUD
    }

def registrar_usuario_con_cara(nombre, idioma, vector_facial):
    """
    Función específica para registrar usuario con reconocimiento facial
    (Wrapper para mantener compatibilidad con el código principal)
    """
    return registrar_usuario(nombre, idioma, vector_facial)

def obtener_datos_visibles_usuario(nombre):
    """
    Retorna los datos del usuario excluyendo el vector facial
    """
    usuario = obtener_usuario(nombre)
    if not usuario:
        print(f"❌ Usuario {nombre} no encontrado")
        return None
    
    datos_visibles = {k: v for k, v in usuario.items() if k != "vector_facial"}
    return datos_visibles

def actualizar_nombre_usuario(nombre_anterior, nombre_nuevo):
    return actualizar_usuario(nombre_actual=nombre_anterior, nuevo_nombre=nombre_nuevo)

def actualizar_idioma_usuario(nombre_usuario, nuevo_idioma):
    return actualizar_usuario(nombre_actual=nombre_usuario, nuevo_idioma=nuevo_idioma)

def actualizar_usuario(nombre_actual, nuevo_nombre=None, nuevo_idioma=None):
    usuarios = cargar_usuarios()
    key_actual = nombre_actual.lower()

    if key_actual not in usuarios:
        print(f"El usuario '{nombre_actual}' no existe.")
        return False

    usuario = usuarios[key_actual]

    # Actualizar nombre
    if nuevo_nombre:
        key_nuevo = nuevo_nombre.lower()
        usuarios[key_nuevo] = usuario
        del usuarios[key_actual]
        usuario = usuarios[key_nuevo]
        usuario["nombre"] = nuevo_nombre
        key_actual = key_nuevo  
    else:
        nuevo_nombre = nombre_actual

    # Actualizar idioma
    if nuevo_idioma:
        usuario["idioma"] = nuevo_idioma
        from datetime import datetime
        usuario["fecha_actualizacion_idioma"] = datetime.now().isoformat()

    guardar_usuarios(usuarios)
    print(f"Usuario '{nuevo_nombre}' actualizado correctamente.")
    return True





