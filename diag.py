import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --- CONFIGURACIÓN ---
directorio_csv = r'C:\Users\LIghtning\Documents\emociones\Sign-Language-Interpreter\CSV_Abecedario'

# --- CARGA AUTOMÁTICA DE DATOS ---
all_features = []
all_labels = []
label_map = {} # Para saber qué número corresponde a qué letra
current_label_index = 0

print("Iniciando carga de datos...")

if not os.path.exists(directorio_csv):
    print(f"Error: El directorio {directorio_csv} no existe.")
    exit()

# Ordenamos los archivos para que las letras estén en orden alfabético
archivos_csv = sorted([f for f in os.listdir(directorio_csv) if f.endswith('.csv')])

for archivo in archivos_csv:
    letra = archivo.split('_')[0]
    ruta_archivo = os.path.join(directorio_csv, archivo)
    
    try:
        datos = np.loadtxt(ruta_archivo, delimiter=',', dtype=str)
        if datos.ndim == 1:
            datos = np.array([datos])
            
        features = datos[:, 1:].astype(float)
        
        # Guardamos las features y creamos las etiquetas numéricas
        all_features.append(features)
        
        # Asignamos un número único a cada letra
        if letra not in label_map:
            label_map[letra] = current_label_index
            current_label_index += 1
            
        # Añadimos las etiquetas para cada muestra
        num_muestras = len(features)
        all_labels.extend([label_map[letra]] * num_muestras)
        
        print(f"✅ Cargado: {num_muestras} muestras para la letra '{letra}'")

    except Exception as e:
        print(f"❌ Error cargando {ruta_archivo}: {e}")

if not all_features:
    print("No se cargaron datos. Saliendo.")
    exit()

# --- PREPARACIÓN DE DATOS ---
X = np.vstack(all_features)
y = np.array(all_labels)

print(f"\nDatos cargados. Forma de X: {X.shape}, Forma de y: {y.shape}")

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducir de 63 dimensiones a 2 para poder graficar
print("Calculando PCA...")
pca = PCA(n_components=12)
X_pca = pca.fit_transform(X_scaled)

# --- CREACIÓN DEL GRÁFICO AUTOMÁTICO ---
num_labels = len(label_map)
# Usamos un mapa de colores para asignar un color único a cada letra
colors = plt.cm.get_cmap('jet', num_labels) 

plt.figure(figsize=(20, 15))

# Invertimos el label_map para poder buscar la letra por su número
letras = {v: k for k, v in label_map.items()}

# Bucle para dibujar cada letra con su color
for i in range(num_labels):
    # Filtramos los puntos que corresponden a la letra actual
    puntos = X_pca[y == i]
    plt.scatter(puntos[:, 0], puntos[:, 1], color=colors(i), label=f'Letra {letras[i]}', alpha=0.7)

plt.title('Visualización de Datos de Señas (Todas las Letras)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(loc='best', ncol=2) # La leyenda se ajusta mejor
plt.grid(True)
print("Mostrando gráfico...")
plt.show()