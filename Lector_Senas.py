import cv2
import mediapipe as mp
import numpy as np
import os
import time
import string
import pickle
from typing import Tuple, List, Dict, Any, Optional

# -----------------------------------------------------------------------------
# SECCIÓN DE CONFIGURACIÓN
# -----------------------------------------------------------------------------

# --- Rutas ---
RUTA_BASE = r'C:\Users\LIghtning\Documents\emociones\Sign-Language-Interpreter'
RUTA_DATOS_CSV = os.path.join(RUTA_BASE, 'CSV_Abecedario')
RUTA_IMAGEN_REFERENCIA = os.path.join(RUTA_BASE, 'abecedario.png')
RUTA_MODELO_IA = os.path.join(RUTA_BASE, 'modelo_ia_senas.pkl')

# --- Parámetros del Modelo de IA ---
MODELO_IA_TIPO = "SVM"  # Elige entre "RandomForest", "SVM", "KNN"
UMBRAL_CONFIANZA_PREDICCION = 0.3  # Requiere 60% de confianza para mostrar una predicción

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4)

# -----------------------------------------------------------------------------
# SECCIÓN DE FUNCIONES DE IA
# -----------------------------------------------------------------------------

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    scikit_learn_disponible = True
except ImportError:
    print("ADVERTENCIA: scikit-learn no está instalado.")
    print("Para usar el reconocimiento por IA, instale con: pip install scikit-learn")
    scikit_learn_disponible = False

def extraer_features(hand_landmarks) -> Optional[np.ndarray]:
    """Convierte los landmarks de la mano en un vector de características normalizado."""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    if landmarks.size == 0:
        return None
        
    origen = landmarks[0].copy()  # Punto de la muñeca
    landmarks_relativos = landmarks - origen
    
    distancia_palma = np.linalg.norm(landmarks_relativos[9])
    if distancia_palma == 0:
        return None
        
    landmarks_normalizados = landmarks_relativos / distancia_palma
    return landmarks_normalizados.flatten()

def cargar_datos_entrenamiento() -> Dict[str, List[np.ndarray]]:
    """Carga todos los archivos CSV de datos y los organiza en un diccionario."""
    datos_por_letra = {}
    if not os.path.exists(RUTA_DATOS_CSV):
        print(f"Error: El directorio de datos {RUTA_DATOS_CSV} no existe.")
        return datos_por_letra

    for archivo in os.listdir(RUTA_DATOS_CSV):
        if archivo.endswith(".csv"):
            try:
                letra = archivo.split('_')[0]
                ruta_archivo = os.path.join(RUTA_DATOS_CSV, archivo)
                datos = np.loadtxt(ruta_archivo, delimiter=',', dtype=str)
                
                if datos.ndim == 1:
                    datos = np.array([datos])
                
                features = datos[:, 1:].astype(float)
                
                if letra not in datos_por_letra:
                    datos_por_letra[letra] = []
                datos_por_letra[letra].extend(features)
            except Exception as e:
                print(f"❌ Error al procesar el archivo {archivo}: {e}")
    
    print("\n--- Resumen de Datos Cargados para Entrenamiento ---")
    for letra, datos in datos_por_letra.items():
        print(f"Letra '{letra}': {len(datos)} muestras.")
    print("--------------------------------------------------\n")
    return datos_por_letra

def entrenar_modelo_ia(datos_entrenamiento: Dict, tipo_modelo: str) -> Tuple[Optional[Any], Optional[Any]]:
    """Entrena un modelo de IA y su escalador a partir de los datos cargados."""
    if not datos_entrenamiento:
        print("No hay datos para entrenar el modelo.")
        return None, None
        
    print(f"Entrenando modelo IA ({tipo_modelo}) con datos de {len(datos_entrenamiento)} letras...")
    
    X, y = [], []
    for letra, datos in datos_entrenamiento.items():
        for muestra in datos:
            X.append(muestra)
            y.append(letra)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if tipo_modelo == "SVM":
        modelo = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    elif tipo_modelo == "KNN":
        modelo = KNeighborsClassifier(n_neighbors=5)
    else:
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    
    modelo.fit(X_train_scaled, y_train)
    
    y_pred = modelo.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Precisión del modelo en datos de prueba: {accuracy:.4f}")
    
    try:
        with open(RUTA_MODELO_IA, 'wb') as f:
            pickle.dump((modelo, scaler, tipo_modelo), f)
        print(f"Modelo IA guardado en: {RUTA_MODELO_IA}")
    except Exception as e:
        print(f"Error al guardar el modelo IA: {e}")
    
    return modelo, scaler

def cargar_modelo_ia() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """Carga un modelo IA y su escalador desde un archivo .pkl."""
    if os.path.exists(RUTA_MODELO_IA):
        try:
            with open(RUTA_MODELO_IA, 'rb') as f:
                modelo, scaler, tipo = pickle.load(f)
            print(f"Modelo IA ({tipo}) cargado desde archivo.")
            return modelo, scaler, tipo
        except Exception as e:
            print(f"Error al cargar modelo IA: {e}")
    return None, None, None

def reconocer_letra(features: np.ndarray, modelo: Any, scaler: Any) -> Optional[str]:
    """Reconoce una letra a partir de las características normalizadas de la mano."""
    if modelo is None or scaler is None or features is None:
        return None
        
    try:
        features_scaled = scaler.transform([features])
        
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(features_scaled)[0]
            max_prob = np.max(probabilidades)
            
            if max_prob < UMBRAL_CONFIANZA_PREDICCION:
                return None # No hay suficiente confianza en la predicción
            
            letra_predicha = modelo.classes_[np.argmax(probabilidades)]
        else:
            letra_predicha = modelo.predict(features_scaled)[0]
            
        return letra_predicha
    except Exception as e:
        print(f"Error durante el reconocimiento: {e}")
        return None

# -----------------------------------------------------------------------------
# SECCIÓN DE FUNCIONES DE UI Y AYUDA
# -----------------------------------------------------------------------------

def cargar_imagen_referencia() -> np.ndarray:
    """Carga la imagen de referencia o crea una por defecto si no la encuentra."""
    if os.path.exists(RUTA_IMAGEN_REFERENCIA):
        imagen = cv2.imread(RUTA_IMAGEN_REFERENCIA)
        if imagen is not None:
            return imagen
            
    print(f"Advertencia: No se encontró la imagen de referencia en {RUTA_IMAGEN_REFERENCIA}")
    img_default = np.ones((600, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img_default, "Imagen de referencia", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img_default, "no encontrada", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img_default

def solicitar_nombre() -> str:
    """Muestra una ventana para que el usuario ingrese su nombre."""
    nombre = ""
    ventana_nombre = "Ingresa tu nombre"
    cv2.namedWindow(ventana_nombre)
    
    while True:
        lienzo = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(lienzo, "Ingresa tu nombre y presiona ENTER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(lienzo, nombre + "|", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(ventana_nombre, lienzo)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 and nombre: # ENTER
            break
        if key == 27: # ESC
            nombre = "Usuario"
            break
        if key == 8: # BACKSPACE
            nombre = nombre[:-1]
        elif key in map(ord, string.ascii_letters + string.digits + ' ') and len(nombre) < 20:
            nombre += chr(key)
            
    cv2.destroyWindow(ventana_nombre)
    return nombre.strip() or "Usuario"

def crear_panel_inferior(ancho: int, texto_a_mostrar: str) -> np.ndarray:
    """Crea el panel inferior de la UI con el texto centrado."""
    altura_panel = 100
    panel = np.zeros((altura_panel, ancho, 3), dtype=np.uint8)
    
    escala_fuente = 1.2
    grosor = 2
    (ancho_texto, alto_texto), _ = cv2.getTextSize(texto_a_mostrar, cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, grosor)
    
    pos_x = (ancho - ancho_texto) // 2
    pos_y = (altura_panel + alto_texto) // 2
    
    cv2.putText(panel, texto_a_mostrar, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, (255, 255, 255), grosor)
    return panel

# -----------------------------------------------------------------------------
# LÓGICA PRINCIPAL DE LA APLICACIÓN
# -----------------------------------------------------------------------------

def main():
    """Función principal que ejecuta la aplicación."""
    global MODELO_IA_TIPO # <-- AÑADE ESTA LÍNEA

    # --- Carga y Entrenamiento del Modelo ---
    modelo_ia, escalador, tipo_cargado = cargar_modelo_ia()
    
    if modelo_ia is None:
        datos_entrenamiento = cargar_datos_entrenamiento()
        if not datos_entrenamiento:
            print("No hay datos para continuar. Saliendo.")
            return
        # Ahora esta línea funcionará porque sabe que MODELO_IA_TIPO es global
        modelo_ia, escalador = entrenar_modelo_ia(datos_entrenamiento, MODELO_IA_TIPO)
    else:
        MODELO_IA_TIPO = tipo_cargado

    # --- Inicialización de la UI ---
    nombre_usuario = solicitar_nombre()
    palabra_actual = ""
    letra_predicha = None
    tiempo_ultimo_espacio = 0
    
    cap = cv2.VideoCapture(0)
    img_ref = cargar_imagen_referencia()
    cv2.namedWindow("Interprete de Lenguaje de Señas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interprete de Lenguaje de Señas", 1920, 1080)

    # --- Bucle Principal ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # --- Detección y Reconocimiento ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extraer_features(hand_landmarks)
                letra_predicha = reconocer_letra(features, modelo_ia, escalador)

        # --- Lógica de Teclado ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and letra_predicha:
            # Evitar agregar espacios muy rápido
            if (time.time() - tiempo_ultimo_espacio) > 0.5:
                palabra_actual += letra_predicha
                tiempo_ultimo_espacio = time.time()
        elif key == ord('r'):
            palabra_actual = ""
        elif key == 8: # Backspace
             palabra_actual = palabra_actual[:-1]
        elif key == 27: # ESC
            break
            
        # --- Composición y Visualización de la UI ---
        alto_frame, ancho_frame = frame.shape[:2]
        img_ref_resized = cv2.resize(img_ref, (ancho_frame // 2, alto_frame))
        
        # Mostrar predicción en el frame
        texto_prediccion = f'Letra: {letra_predicha or "N/A"}'
        cv2.putText(frame, texto_prediccion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        interfaz_superior = np.hstack([frame, img_ref_resized])
        ancho_total = interfaz_superior.shape[1]
        
        texto_panel = f"{nombre_usuario} dice: {palabra_actual}"
        panel_inferior = crear_panel_inferior(ancho_total, texto_panel)
        
        interfaz_completa = np.vstack([interfaz_superior, panel_inferior])
        cv2.imshow("Interprete de Lenguaje de Señas", interfaz_completa)

    # --- Limpieza Final ---
    cap.release()
    cv2.destroyAllWindows()
    print("\nPrograma finalizado.")

if __name__ == "__main__":
    main()