import cv2
import numpy as np

def region_inferior(fotograma):
    '''
    Devuelve el cuarto inferior del fotograma.
    Zona donde presuntamente suelen estar los subtítulos.
    '''
    alto, _, _ = fotograma.shape
    fotograma_inferior = fotograma[int(alto/4)*3:,:]
    return fotograma_inferior

def procesar_fotograma(fotograma):
    # Region de interés
    imagen_bgr = region_inferior(fotograma) # Cuarto inferior del fotograma
    # Reducción de ruido
    imagen_bgr = cv2.GaussianBlur(imagen_bgr,(3,3),0) # Desenfoque Gaussiano
    imagen_bgr = cv2.medianBlur(imagen_bgr,3) # Filtro de mediana (no lineal)
    # Promedio de canales: B, G, R, S, V, G (Se descarta el H)
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV) # Imagen en formato HSV
    imagen_gs = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY) # imagen en escala de grises
    imagen_bgr = np.float32(imagen_bgr) # Conversión del dtype
    imagen_hsv = np.float32(imagen_hsv) # Conversión del dtype
    imagen_gs = np.float32(imagen_gs) # Conversión del dtype
    promedio = (imagen_bgr[:,:,0] + imagen_bgr[:,:,1] + imagen_bgr[:,:,2] + imagen_hsv[:,:,1] + imagen_hsv[:,:,2] + imagen_gs)/6 # Promedio de 6 imágenes
    promedio = np.uint8(promedio) # Conversión del dtype
    # Detección de bordes
    kernel = -np.ones((3,3),np.float32) # Kernel para la detección de bordes
    kernel[1,1] = 7
    promedio = cv2.filter2D(promedio,-1,kernel) # Filtrado usando el kernel
    # Binarización
    promedio = cv2.threshold(promedio, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1] # En la posición 1 está la imagen
    # Redimensionamiento a un cuarto del tamaño original
    promedio = cv2.resize(promedio, (int(promedio.shape[1]/4),int(promedio.shape[0]/4))) # (Columnas,filas)

    # cv2.imshow("Fotograma",promedio)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return promedio

fotograma_ruta = "fotograma.png"
fotograma = cv2.imread(fotograma_ruta, cv2.IMREAD_COLOR)

procesar_fotograma(fotograma)


