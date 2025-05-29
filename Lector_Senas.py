import cv2
import mediapipe as mp
import numpy as np
import os
import time
import string
import pickle
from typing import Tuple, Optional

# Importaciones para el modelo de IA
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    scikit_learn_disponible = True
except ImportError:
    print("ADVERTENCIA: scikit-learn no está instalado. Se usará reconocimiento basado en distancias.")
    print("Para usar reconocimiento basado en IA, instale scikit-learn con: pip install scikit-learn")
    scikit_learn_disponible = False
    
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

ruta_modelos = r'C:\Users\Lightning\Documents\Sign-Language-Interpreter\CSV_Abecedario'
ruta_imagen_referencia = r'C:\Users\Lightning\Documents\Sign-Language-Interpreter\abecedario.png'
ruta_modelo_ia = r'C:\Users\Lightning\Documents\Sign-Language-Interpreter\modelo_ia_senas.pkl'

# Variables globales para el modelo de IA
modelo_ia = None
escalador = None
modelo_ia_tipo = "RandomForest"  

# Cargar modelos de letras
def cargar_modelos():
    modelos = {}
    try:
        # Verificar si el directorio existe
        if not os.path.exists(ruta_modelos):
            print(f"Error: El directorio {ruta_modelos} no existe")
            return modelos
            
        for archivo in os.listdir(ruta_modelos):
            if archivo.endswith(".csv"):
                try:
                    letra = archivo.split('_')[0]  
                    ruta_archivo = os.path.join(ruta_modelos, archivo)
                    modelos[letra] = np.loadtxt(ruta_archivo, delimiter=',', usecols=range(1, 64))
                    print(f"Modelo cargado: {letra}")
                except Exception as e:
                    print(f"Error al cargar el modelo {archivo}: {str(e)}")
    except Exception as e:
        print(f"Error general al cargar modelos: {str(e)}")
    
    if not modelos:
        print("Advertencia: No se cargó ningún modelo de letra")
    else:
        print(f"Total de modelos cargados: {len(modelos)}")
        
    return modelos

# Función para entrenar modelo de IA basado en los datos CSV
def entrenar_modelo_ia(modelos, tipo_modelo="RandomForest"):
    """
    Entrena un modelo de IA utilizando los datos CSV cargados.
    
    Args:
        modelos: Diccionario con los datos de las letras cargados de los CSV
        tipo_modelo: Tipo de modelo a entrenar ("RandomForest", "SVM", "KNN")
        
    Returns:
        tuple: Modelo entrenado y escalador de características
    """
    if not scikit_learn_disponible:
        print("No se puede entrenar modelo IA: scikit-learn no está instalado")
        return None, None
        
    if not modelos:
        print("No hay datos para entrenar el modelo IA")
        return None, None
        
    print(f"Entrenando modelo IA ({tipo_modelo}) con datos de {len(modelos)} letras...")
    
    # Preparar datos para entrenamiento
    X = []  # Características (landmarks)
    y = []  # Etiquetas (letras)
    
    for letra, datos in modelos.items():
        for muestra in datos:
            X.append(muestra)
            y.append(letra)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Seleccionar y entrenar el modelo según el tipo
    if tipo_modelo == "SVM":
        modelo = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    elif tipo_modelo == "KNN":
        modelo = KNeighborsClassifier(n_neighbors=5)
    else:  # Por defecto, RandomForest
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Entrenar el modelo
    modelo.fit(X_train_scaled, y_train)
    
    # Evaluar el modelo
    y_pred = modelo.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo {tipo_modelo}: {accuracy:.4f}")
    
    # Guardar el modelo entrenado
    try:
        with open(ruta_modelo_ia, 'wb') as archivo:
            pickle.dump((modelo, scaler, tipo_modelo), archivo)
        print(f"Modelo IA guardado en: {ruta_modelo_ia}")
    except Exception as e:
        print(f"Error al guardar el modelo IA: {e}")
    
    return modelo, scaler

# Función para cargar el modelo IA previamente entrenado
def cargar_modelo_ia():
    """
    Carga un modelo IA previamente entrenado.
    
    Returns:
        tuple: Modelo, escalador y tipo de modelo, o (None, None, None) si no se puede cargar
    """
    if not scikit_learn_disponible:
        return None, None, None
        
    try:
        if os.path.exists(ruta_modelo_ia):
            with open(ruta_modelo_ia, 'rb') as archivo:
                modelo, scaler, tipo_modelo = pickle.load(archivo)
            print(f"Modelo IA ({tipo_modelo}) cargado desde: {ruta_modelo_ia}")
            return modelo, scaler, tipo_modelo
        else:
            print("No se encontró un modelo IA guardado. Se entrenará uno nuevo.")
            return None, None, None
    except Exception as e:
        print(f"Error al cargar el modelo IA: {e}")
        return None, None, None

modelos = cargar_modelos()

# Cargar o entrenar el modelo IA
if scikit_learn_disponible:
    modelo_ia, escalador, tipo_cargado = cargar_modelo_ia()
    
    # Si no hay modelo guardado o no se pudo cargar, entrenar uno nuevo
    if modelo_ia is None:
        modelo_ia, escalador = entrenar_modelo_ia(modelos, modelo_ia_tipo)
    else:
        modelo_ia_tipo = tipo_cargado
else:
    print("Usando reconocimiento basado en distancias (sin IA)")
# Función para cargar la imagen de referencia
def cargar_imagen_referencia():
    # Comprobar si la imagen existe
    if os.path.exists(ruta_imagen_referencia):
        imagen = cv2.imread(ruta_imagen_referencia)
        if imagen is not None:
            print(f"Imagen de referencia cargada desde: {ruta_imagen_referencia}")
            return imagen
    
    # Si no existe, crear una imagen con un mensaje
    print(f"No se encontró la imagen de referencia en: {ruta_imagen_referencia}")
    imagen_default = np.ones((600, 400, 3), dtype=np.uint8) * 255  # Fondo blanco
    cv2.putText(imagen_default, "Imagen de referencia", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imagen_default, "no encontrada", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(imagen_default, "Coloca una imagen en:", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(imagen_default, f"{ruta_imagen_referencia}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(imagen_default, "con todos los símbolos", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(imagen_default, "del lenguaje de señas", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return imagen_default

# Cargar imagen de referencia
imagen_referencia = cargar_imagen_referencia()

# Función para redimensionar imagen manteniendo relación de aspecto
def redimensionar_imagen(imagen, alto_objetivo):
    alto, ancho = imagen.shape[:2]
    ratio = alto_objetivo / alto
    nuevo_ancho = int(ancho * ratio)
    return cv2.resize(imagen, (nuevo_ancho, alto_objetivo))

# Constantes para propiedades de ventanas en OpenCV (para compatibilidad)
WND_PROP_WIDTH = 0   # Valor numérico para cv2.WND_PROP_WIDTH
WND_PROP_HEIGHT = 1  # Valor numérico para cv2.WND_PROP_HEIGHT

# Función segura para obtener propiedades de ventana
def obtener_tamano_ventana(nombre_ventana, ancho_por_defecto, alto_por_defecto):
    """Obtiene el tamaño de una ventana de forma segura con valores por defecto"""
    try:
        ancho = int(cv2.getWindowProperty(nombre_ventana, WND_PROP_WIDTH))
        alto = int(cv2.getWindowProperty(nombre_ventana, WND_PROP_HEIGHT))
        
        # Verificar si los valores son válidos (mayores que 1)
        if ancho > 1 and alto > 1:
            return ancho, alto
        else:
            return ancho_por_defecto, alto_por_defecto
    except Exception as e:
        print(f"Advertencia: No se pudo obtener el tamaño de la ventana: {e}")
        return ancho_por_defecto, alto_por_defecto

# Clase para manejar el escalado de la interfaz
class EscaladorInterfaz:
    def __init__(self, ancho_base=1280, alto_base=720):
        self.ancho_base = ancho_base
        self.alto_base = alto_base
        self.ancho_actual = ancho_base
        self.alto_actual = alto_base
        self.factor_escala_x = 1.0
        self.factor_escala_y = 1.0
        self.ultima_actualizacion = 0
        
    def actualizar_dimensiones(self, ancho: int, alto: int) -> bool:
        """Actualiza las dimensiones de la ventana y calcula factores de escala"""
        # Solo actualizar si ha cambiado más de 10 píxeles para evitar actualizaciones constantes
        if (abs(ancho - self.ancho_actual) > 10 or abs(alto - self.alto_actual) > 10 or 
            time.time() - self.ultima_actualizacion > 1.0):
            self.ancho_actual = ancho
            self.alto_actual = alto
            self.factor_escala_x = ancho / self.ancho_base
            self.factor_escala_y = alto / self.alto_base
            self.ultima_actualizacion = time.time()
            return True
        return False
        
    def escalar_tamano(self, ancho: int, alto: int) -> Tuple[int, int]:
        """Escala un tamaño según los factores actuales"""
        return (int(ancho * self.factor_escala_x), int(alto * self.factor_escala_y))
        
    def escalar_valor(self, valor: float) -> int:
        """Escala un valor según un promedio de los factores de escala"""
        factor_promedio = (self.factor_escala_x + self.factor_escala_y) / 2
        return max(1, int(valor * factor_promedio))
        
    def escalar_coordenadas(self, x: int, y: int) -> Tuple[int, int]:
        """Escala coordenadas x,y según los factores actuales"""
        return (int(x * self.factor_escala_x), int(y * self.factor_escala_y))
        
    def obtener_escala_fuente(self, escala_base: float) -> float:
        """Obtiene una escala de fuente ajustada"""
        factor_promedio = (self.factor_escala_x + self.factor_escala_y) / 2
        return escala_base * factor_promedio

# Función para agregar borde a una imagen
def agregar_borde(imagen, color=(0, 120, 255), grosor=3):
    h, w = imagen.shape[:2]
    bordeada = imagen.copy()
    cv2.rectangle(bordeada, (0, 0), (w-1, h-1), color, grosor)
    return bordeada

def reconocer_letra(landmarks, modelos, umbral=0.275):
    """
    Reconoce una letra a partir de los landmarks de la mano.
    
    Si el modelo de IA está disponible, utiliza ese para la clasificación.
    Si no, utiliza el método de distancia euclidiana tradicional.
    
    Args:
        landmarks: Array de landmarks de la mano
        modelos: Diccionario con los datos de referencia de las letras
        umbral: Umbral para el método de distancia (solo usado si no hay IA)
        
    Returns:
        str: Letra reconocida o None si no se detectó ninguna
    """
    global modelo_ia, escalador
    
    # Usar el modelo de IA si está disponible
    if scikit_learn_disponible and modelo_ia is not None and escalador is not None:
        try:
            # Preprocedr los landmarks
            landmarks_escalados = escalador.transform([landmarks])
            
            # Predecir letra con probabilidades
            # Obtenemos ambas la predicción y las probabilidades de cada clase
            letra_predicha = modelo_ia.predict(landmarks_escalados)[0]
            
            # Obtener la probabilidad de la predicción
            if hasattr(modelo_ia, 'predict_proba'):
                probabilidades = modelo_ia.predict_proba(landmarks_escalados)[0]
                max_prob = np.max(probabilidades)
                
                # Verificar si la probabilidad supera un umbral mínimo
                if max_prob < 0.6:  # Umbral de confianza (ajustable)
                    return None
            
            return letra_predicha
            
        except Exception as e:
            print(f"Error al usar modelo IA: {e}")
            # Si hay error con el modelo IA, intentar con el método tradicional
    
    # Método tradicional de distancia euclidiana (fallback)
    letra_reconocida, distancia_minima = None, float('inf')
    
    for letra, modelo in modelos.items():
        distancia = np.min(np.linalg.norm(modelo - landmarks, axis=1))
        if distancia < distancia_minima:
            distancia_minima, letra_reconocida = distancia, letra

    return letra_reconocida if distancia_minima < umbral else None

# Variables de letra
# Función para solicitar el nombre del usuario
def solicitar_nombre():
    nombre = ""
    # Crear ventana redimensionable
    cv2.namedWindow("Ingresa tu nombre", cv2.WINDOW_NORMAL)
    # Dimensiones iniciales de la ventana
    ancho_ventana, alto_ventana = 600, 300
    cv2.resizeWindow("Ingresa tu nombre", ancho_ventana, alto_ventana)
    
    # Crear instancia del escalador
    escalador = EscaladorInterfaz(ancho_base=600, alto_base=300)
    
    ventana_nombre = np.zeros((alto_ventana, ancho_ventana, 3), dtype=np.uint8)
    
    while True:
        # Crear una copia limpia
        ventana = ventana_nombre.copy()
        
        # Obtener dimensiones actuales de la ventana de forma segura
        ancho_actual, alto_actual = obtener_tamano_ventana("Ingresa tu nombre", ancho_ventana, alto_ventana)
        
        # Actualizar dimensiones del escalador si es necesario
        if escalador.actualizar_dimensiones(ancho_actual, alto_actual):
            # Redimensionar el buffer de la ventana si hay un cambio significativo
            alto_ventana, ancho_ventana = escalador.escalar_tamano(300, 600)
            ventana = np.zeros((alto_ventana, ancho_ventana, 3), dtype=np.uint8)
        
        # Posiciones escaladas para el texto
        pos_titulo = escalador.escalar_coordenadas(20, 40)
        pos_instruccion = escalador.escalar_coordenadas(20, 80)
        pos_nombre = escalador.escalar_coordenadas(20, 130)
        pos_enter = escalador.escalar_coordenadas(20, 200)
        pos_esc = escalador.escalar_coordenadas(20, 230)
        
        # Escalas de fuente ajustadas
        escala_titulo = escalador.obtener_escala_fuente(0.8)
        escala_instruccion = escalador.obtener_escala_fuente(0.7)
        escala_nombre = escalador.obtener_escala_fuente(1.0)
        escala_teclas = escalador.obtener_escala_fuente(0.7)
        
        # Grosor de línea escalado
        grosor_titulo = escalador.escalar_valor(2)
        grosor_nombre = escalador.escalar_valor(2)
        grosor_normal = escalador.escalar_valor(1)
        
        # Dibujar título e instrucciones con valores escalados
        cv2.putText(ventana, "Bienvenido al Interprete de Lenguaje de Senas", pos_titulo, 
                    cv2.FONT_HERSHEY_SIMPLEX, escala_titulo, (255, 255, 255), grosor_titulo)
        cv2.putText(ventana, "Por favor, ingresa tu nombre:", pos_instruccion, 
                    cv2.FONT_HERSHEY_SIMPLEX, escala_instruccion, (255, 255, 255), grosor_normal)
        cv2.putText(ventana, nombre + "|", pos_nombre, 
                    cv2.FONT_HERSHEY_SIMPLEX, escala_nombre, (0, 255, 0), grosor_nombre)
        cv2.putText(ventana, "Presiona ENTER para continuar", pos_enter, 
                    cv2.FONT_HERSHEY_SIMPLEX, escala_teclas, (200, 200, 200), grosor_normal)
        cv2.putText(ventana, "Presiona ESC para usar 'Usuario'", pos_esc, 
                    cv2.FONT_HERSHEY_SIMPLEX, escala_teclas, (200, 200, 200), grosor_normal)
        
        # Mostrar ventana
        cv2.imshow("Ingresa tu nombre", ventana)
        
        # Capturar tecla
        key = cv2.waitKey(1) & 0xFF
        
        # Si presiona ENTER y hay un nombre, continuar
        if key == 13 and nombre.strip():  # ENTER
            break
            
        # Si presiona ESC, usar 'Usuario' por defecto
        if key == 27:  # ESC
            nombre = "Usuario"
            break
            
        # Borrar último carácter con BACKSPACE
        if key == 8 and len(nombre) > 0:  # BACKSPACE
            nombre = nombre[:-1]
            
        # Agregar caracteres permitidos
        if key in map(ord, string.ascii_letters + string.digits + ' áéíóúÁÉÍÓÚñÑ'):
            # Limitar la longitud del nombre a 20 caracteres
            if len(nombre) < 20:
                nombre += chr(key)
    
    cv2.destroyWindow("Ingresa tu nombre")
    return nombre if nombre.strip() else "Usuario"

# Función para crear el panel inferior
def crear_panel_inferior(nombre, palabra, ancho_total, escalador):
    # Altura base para el panel inferior (será escalada)
    altura_panel_base = 100
    altura_panel = escalador.escalar_valor(altura_panel_base)
    
    # Crear panel con fondo oscuro
    panel = np.zeros((altura_panel, ancho_total, 3), dtype=np.uint8)
    # Crear el mensaje formateado
    mensaje = f"{nombre} dice: {palabra}"
    
    # Calcular tamaño de texto que se ajuste al ancho del panel
    grosor_texto = escalador.escalar_valor(2)
    escala_base = 1.0
    
    # Ajustar escala de fuente según longitud del mensaje y factor de escala
    escala = escalador.obtener_escala_fuente(escala_base - (len(mensaje) * 0.01))
    escala = max(0.5, min(escala, 1.5 * escalador.factor_escala_y))  # Ajustar límites
    
    # Obtener dimensiones del texto
    (ancho_texto, alto_texto), _ = cv2.getTextSize(mensaje, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor_texto)
    
    # Calcular posición para centrar el texto
    pos_x = (ancho_total - ancho_texto) // 2
    pos_y = (altura_panel + alto_texto) // 2
    
    # Agregar padding visual (rectángulo con borde redondeado para el fondo del texto)
    padding_x = escalador.escalar_valor(20)
    padding_y = escalador.escalar_valor(10)
    
    # Definir los puntos p1 y p2 para el rectángulo de fondo
    p1 = (pos_x - padding_x, pos_y - alto_texto - padding_y)
    p2 = (pos_x + ancho_texto + padding_x, pos_y + padding_y)
    
    # Dibujar un rectángulo como fondo del texto
    cv2.rectangle(panel, p1, p2, (40, 40, 40), -1)
    cv2.rectangle(panel, p1, p2, (100, 100, 100), 1)
    
    # Dibujar el texto
    cv2.putText(panel, mensaje, (pos_x, pos_y), 
                cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), grosor_texto)
    
    return panel

palabra = ""  
letra_detectada = None  # Letra reconocida en tiempo real
feedback_activo = False  # Variable para controlar el feedback visual
tiempo_feedback = 0  # Variable para controlar la duración del feedback
palabra_reiniciada = False  # Variable para indicar reinicio de palabra
tiempo_reinicio = 0  # Variable para controlar la duración del mensaje de reinicio

# Solicitar nombre al inicio
nombre_usuario = solicitar_nombre()

# Crear ventana redimensionable
cv2.namedWindow("Interprete de Lenguaje de Señas", cv2.WINDOW_NORMAL)
# Establecer tamaño inicial de la ventana
ancho_ventana_inicial, alto_ventana_inicial = 1280, 720
cv2.resizeWindow("Interprete de Lenguaje de Señas", ancho_ventana_inicial, alto_ventana_inicial)

# Crear instancia del escalador para la interfaz principal
escalador_interfaz = EscaladorInterfaz(ancho_base=1280, alto_base=720)

# Captura en tiempo real
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    letra_detectada = None  

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            letra_detectada = reconocer_letra(landmarks, modelos)

    # Definir colores para el texto
    color_palabra = (0, 255, 0)  # Verde por defecto
    color_letra = (0, 255, 0)    # Verde por defecto
    color_instrucciones = (255, 255, 255)  # Blanco
    
    # Comprobar si el feedback está activo y no ha expirado
    if feedback_activo and time.time() - tiempo_feedback < 1.0:
        color_palabra = (0, 0, 255)  # Cambiar a rojo para feedback
    else:
        feedback_activo = False
        
    # Comprobar si el mensaje de reinicio está activo
    if palabra_reiniciada and time.time() - tiempo_reinicio < 1.5:
        escala_reinicio = escalador_interfaz.obtener_escala_fuente(1.0)
        grosor_reinicio = escalador_interfaz.escalar_valor(2)
        pos_reinicio = escalador_interfaz.escalar_coordenadas(10, 150)
        cv2.putText(frame, "Palabra reiniciada", pos_reinicio, 
                    cv2.FONT_HERSHEY_SIMPLEX, escala_reinicio, (0, 0, 255), grosor_reinicio)
    else:
        palabra_reiniciada = False
    
    # Mostrar letra detectada (ya no mostramos la palabra aquí)
    escala_letra = escalador_interfaz.obtener_escala_fuente(1.0)
    grosor_letra = escalador_interfaz.escalar_valor(2)
    pos_letra = escalador_interfaz.escalar_coordenadas(10, 50)
    cv2.putText(frame, f'Letra detectada: {letra_detectada or ""}', pos_letra, 
                cv2.FONT_HERSHEY_SIMPLEX, escala_letra, color_letra, grosor_letra)
    
    # Mostrar instrucciones
    escala_instrucciones = escalador_interfaz.obtener_escala_fuente(0.6)
    grosor_instrucciones = escalador_interfaz.escalar_valor(1)
    pos_instrucciones = escalador_interfaz.escalar_coordenadas(10, frame.shape[0] - 20)
    cv2.putText(frame, "Espacio: Agregar letra  |  R: Reiniciar palabra  |  ESC: Salir", 
                pos_instrucciones, cv2.FONT_HERSHEY_SIMPLEX, escala_instrucciones, 
                color_instrucciones, grosor_instrucciones)

    key = cv2.waitKey(1) & 0xFF

    # Si presiona espacio, agrega la letra detectada
    if key == ord(' ') and letra_detectada:
        palabra += letra_detectada  # Agregar la letra a la palabra
        # Activar feedback visual
        feedback_activo = True
        tiempo_feedback = time.time()
    
    # Si presiona 'r', reinicia la palabra
    if key == ord('r'):
        palabra = ""  # Reiniciar la palabra
        palabra_reiniciada = True
        tiempo_reinicio = time.time()

    # Salir con ESC
    if key == 27:
        break

    # Preparar la interfaz dividida
    # Obtener dimensiones actuales de la ventana de forma segura
    ancho_ventana_actual, alto_ventana_actual = obtener_tamano_ventana(
        "Interprete de Lenguaje de Señas", ancho_ventana_inicial, alto_ventana_inicial)
    
    # Actualizar escalador de interfaz
    escalador_interfaz.actualizar_dimensiones(ancho_ventana_actual, alto_ventana_actual)

    # Obtener dimensiones del frame
    alto_frame, ancho_frame = frame.shape[:2]
    
    # Redimensionar la imagen de referencia para que tenga la misma altura que el frame
    img_ref_resized = redimensionar_imagen(imagen_referencia, alto_frame)
    
    # Agregar bordes a ambas imágenes
    frame_con_borde = agregar_borde(frame, color=(120, 120, 120), grosor=2)
    img_ref_con_borde = agregar_borde(img_ref_resized, color=(120, 120, 120), grosor=2)
    
    # Crear una separación entre las imágenes con ancho escalable
    ancho_separador = escalador_interfaz.escalar_valor(10)
    separador = np.ones((alto_frame, ancho_separador, 3), dtype=np.uint8) * 200  # Gris claro
    
    # Combinar horizontalmente: frame | separador | imagen de referencia
    interfaz_superior = np.hstack([frame_con_borde, separador, img_ref_con_borde])
    
    # Calcular el ancho total de la interfaz
    ancho_total = interfaz_superior.shape[1]
    
    # Crear panel inferior con el nombre y la palabra
    # Añadir indicador del tipo de reconocimiento usado
    if scikit_learn_disponible and modelo_ia is not None:
        info_panel = f"{nombre_usuario} dice: {palabra} [IA: {modelo_ia_tipo}]"
    else:
        info_panel = f"{nombre_usuario} dice: {palabra}"
    panel_inferior = crear_panel_inferior(nombre_usuario, palabra, ancho_total, escalador_interfaz)
    
    # Crear un separador horizontal con altura escalable
    altura_separador = escalador_interfaz.escalar_valor(5)
    separador_horizontal = np.ones((altura_separador, ancho_total, 3), dtype=np.uint8) * 80  # Gris oscuro
    
    # Combinar verticalmente la interfaz completa
    interfaz_completa = np.vstack([interfaz_superior, separador_horizontal, panel_inferior])
    
    # Mostrar la interfaz completa
    cv2.imshow("Interprete de Lenguaje de Señas", interfaz_completa)
    
# Liberar Recursos
cap.release()
cv2.destroyAllWindows()

# Mensaje informativo al cerrar el programa
print("\nPrograma finalizado correctamente.")
print(f"Para agregar una imagen de referencia, coloca un archivo en: {ruta_imagen_referencia}")
if scikit_learn_disponible:
    print(f"Modelo IA usado: {modelo_ia_tipo if modelo_ia else 'Ninguno'}")
    print(f"Ruta del modelo IA: {ruta_modelo_ia}")
else:
    print("Para usar reconocimiento basado en IA, instale scikit-learn: pip install scikit-learn")
