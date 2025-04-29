
# Intérprete de Lengua de Señas

Este proyecto tiene como objetivo desarrollar un intérprete de lengua de señas utilizando Python, OpenCV y Mediapipe. El sistema reconoce las letras del abecedario en lenguaje de señas a través de una cámara en tiempo real, y puede ser entrenado utilizando diferentes modelos de IA para mejorar la precisión.

## Tecnologías utilizadas

- **Python 3.x**  
- **OpenCV**: para captura de video y procesamiento de imágenes.  
- **Mediapipe**: para el seguimiento de las manos y reconocimiento de gestos.  
- **scikit-learn** (opcional): para entrenar y usar modelos de clasificación como Random Forest, SVM y KNN.  
- **NumPy**: para manipulación de matrices y arrays.  
- **Pickle**: para guardar y cargar modelos entrenados.

## Características

- **Reconocimiento de señas en tiempo real**: Detecta y traduce letras del abecedario utilizando un modelo entrenado.  
- **Entrenamiento personalizado**: Puedes entrenar tu propio modelo utilizando datos en formato CSV o cargar un modelo previamente entrenado.  
- **Soporte para múltiples modelos de IA**: Se incluyen modelos como Random Forest, SVM y KNN, con la posibilidad de elegir el que mejor se ajuste a tus necesidades.  
- **Imagen de referencia**: Utiliza una imagen de referencia para mostrar el alfabeto en señas mientras se realizan las predicciones.

## Instalación

1. Clona este repositorio en tu máquina local:

   ```bash
   git clone https://github.com/tu_usuario/sign-language-interpreter.git
   cd sign-language-interpreter
   ```

2. Instala las dependencias necesarias:

   ```bash
   pip install opencv-python mediapipe scikit-learn numpy
   ```

3. Si no tienes scikit-learn, el sistema usará un método de reconocimiento basado en distancias.

## Uso

### Cargar y entrenar el modelo de IA

El proyecto carga los modelos de letras en formato CSV desde una carpeta específica. Si no existe un modelo preentrenado, el sistema lo entrenará automáticamente.

1. **Entrenamiento de un modelo**:  
   Para entrenar un modelo, asegúrate de que los archivos CSV estén en la ruta correcta y ejecuta el script. El modelo se guardará en un archivo pickle para uso posterior.

2. **Uso de un modelo entrenado**:  
   El sistema cargará automáticamente el modelo entrenado desde el archivo pickle y comenzará a reconocer las letras en tiempo real.

### Reconocimiento de señas

El sistema capturará el video desde tu cámara web, detectará las manos y realizará la predicción de la letra que estás mostrando.

```python
import cv2
import mediapipe as mp

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
```

### Cargar una imagen de referencia

Si deseas cargar una imagen con los símbolos del abecedario en señas, colócala en la ruta indicada o el sistema generará una imagen predeterminada.

```python
ruta_imagen_referencia = "ruta/a/tu/imagen.png"
```

## Contribuciones

Las contribuciones son bienvenidas. Si encuentras algún error o tienes sugerencias de mejora, por favor abre un "issue" o envía un "pull request". 

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
