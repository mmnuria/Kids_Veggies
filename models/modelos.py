import modules.cuia as cuia
import numpy as np

def crear_modelo_pera():
    modelo = cuia.modeloGLTF('media/pera.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_cebolleta():
    modelo = cuia.modeloGLTF('media/cebolleta.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_cebolla():
    modelo = cuia.modeloGLTF('media/cebolla.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_lechuga():
    modelo = cuia.modeloGLTF('media/lechuga.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_limon():
    modelo = cuia.modeloGLTF('media/limon.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_pimiento_rojo():
    modelo = cuia.modeloGLTF('media/pimientoRojo.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_pimiento_verde():
    modelo = cuia.modeloGLTF('media/pimientoVerde.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_uvas():
    modelo = cuia.modeloGLTF('media/uvas.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

def crear_modelo_zanahoria():
    """Crear modelo 3D de zanahoria"""
    modelo = cuia.modeloGLTF('media/zanahoria.glb')
    modelo.rotar((np.pi / 2.0, 0, 0))
    modelo.escalar(0.15)
    modelo.flotar()
    animaciones = modelo.animaciones()
    if animaciones:
        modelo.animar(animaciones[0])
    return modelo

# Diccionario de modelos con sus nombres y respuestas correctas
MODELOS_FRUTAS_VERDURAS = {
    0: {
        'crear_modelo': crear_modelo_pera,
        'nombre': 'pera',
        'respuesta_correcta': 'pera',
        'tipo': 'fruta'
    },
    1: {
        'crear_modelo': crear_modelo_cebolleta,
        'nombre': 'cebolleta',
        'respuesta_correcta': 'cebolleta',
        'tipo': 'verdura'
    },
    2: {
        'crear_modelo': crear_modelo_cebolla,
        'nombre': 'cebolla',
        'respuesta_correcta': 'cebolla',
        'tipo': 'verdura'
    },
    3: {
        'crear_modelo': crear_modelo_lechuga,
        'nombre': 'lechuga',
        'respuesta_correcta': 'lechuga',
        'tipo': 'verdura'
    },
    4: {
        'crear_modelo': crear_modelo_limon,
        'nombre': 'limón',
        'respuesta_correcta': 'limon',
        'tipo': 'fruta'
    },
    5: {
        'crear_modelo': crear_modelo_pimiento_rojo,
        'nombre': 'pimiento rojo',
        'respuesta_correcta': 'pimiento rojo',
        'tipo': 'verdura'
    },
    6: {
        'crear_modelo': crear_modelo_pimiento_verde,
        'nombre': 'pimiento verde',
        'respuesta_correcta': 'pimiento verde',
        'tipo': 'verdura'
    },
    7: {
        'crear_modelo': crear_modelo_uvas,
        'nombre': 'uvas',
        'respuesta_correcta': 'uvas',
        'tipo': 'fruta'
    },
    8: {
        'crear_modelo': crear_modelo_zanahoria,
        'nombre': 'zanahoria',
        'respuesta_correcta': 'zanahoria',
        'tipo': 'verdura'
    }
}

def crear_modelo_por_id(marker_id):
    """
    Crear modelo 3D según el ID del marcador ArUco detectado
    """
    if marker_id in MODELOS_FRUTAS_VERDURAS:
        return MODELOS_FRUTAS_VERDURAS[marker_id]['crear_modelo']()
    else:
        # Por defecto, mostrar lechuga si el ID no está en el diccionario
        return crear_modelo_lechuga()

def obtener_info_modelo(marker_id):
    """
    Obtener información del modelo según el ID del marcador
    """
    if marker_id in MODELOS_FRUTAS_VERDURAS:
        return MODELOS_FRUTAS_VERDURAS[marker_id]
    else:
        return MODELOS_FRUTAS_VERDURAS[3]