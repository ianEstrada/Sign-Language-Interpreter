import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# --- Función de Extracción de Características ---
def extraer_features(hand_landmarks):
    """Convierte los landmarks de la mano en un vector de características normalizado."""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    origen = landmarks[0]
    landmarks_relativos = landmarks - origen
    distancia_palma = np.linalg.norm(landmarks_relativos[9])
    if distancia_palma == 0:
        return None
    landmarks_normalizados = landmarks_relativos / distancia_palma
    return landmarks_normalizados.flatten()

# --- Configuración de MediaPipe ---

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Configuración del Script ---
# Pedir al usuario la letra a capturar
letra_actual = input("Ingresa la letra que vas a capturar (ej. A, B, C...): ").upper()
if not letra_actual or len(letra_actual) > 1:
    print("Entrada inválida. Saliendo.")
    exit()

# Crear el nombre del archivo dinámicamente
directorio_csv = r'C:\Users\LIghtning\Documents\emociones\Sign-Language-Interpreter\CSV_Abecedario'
if not os.path.exists(directorio_csv):
    os.makedirs(directorio_csv) # Crea el directorio si no existe

ruta_archivo = os.path.join(directorio_csv, f'{letra_actual}_Models.csv')
print(f"Guardando datos para la letra '{letra_actual}' en el archivo: {ruta_archivo}")



# --- Captura de Video ---
cap = cv2.VideoCapture(0)
contador_muestras = 0

# Bucle principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Invertir la imagen para efecto espejo y mejorar la usabilidad
    frame = cv2.flip(frame, 1)
    
    # Procesamiento con MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Dibujar la interfaz en el frame
    cv2.putText(frame, f"Capturando letra: {letra_actual}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Muestras guardadas: {contador_muestras}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Presiona 'S' para guardar muestra", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'Q' para salir", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        # Asumimos que solo nos interesa la primera mano detectada
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Esperar la tecla para guardar
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            features = extraer_features(hand_landmarks)
            
            if features is not None:
                # Crear la fila para el CSV con la etiqueta primero
                fila_csv = [letra_actual] + list(features)
                
                # Guardar en el archivo CSV
                with open(ruta_archivo, mode='a', newline='') as archivo:
                    writer = csv.writer(archivo)
                    writer.writerow(fila_csv)
                
                contador_muestras += 1
                print(f"Muestra {contador_muestras} para la letra '{letra_actual}' guardada.")
    else:
        # Si no hay mano, solo procesamos la tecla para salir
        key = cv2.waitKey(1) & 0xFF

    cv2.imshow("Captura de Datos para IA", frame)

    if 'key' in locals() and key == ord('q'):
        break

# Liberar Recursos
cap.release()
cv2.destroyAllWindows()
print(f"\nCaptura finalizada. Se guardaron {contador_muestras} muestras en {ruta_archivo}.")