'''
En este archivo se implementan las funciones necesarias para obtener los datos de entrada a la red (imágenes/fotogramas)
a partir de archivos de video MKV con subtítulos flotantes.
'''

import os # Para navegar por los directorios del sistema
import subprocess # Para ejecutar comandos de consola
import sys
import numpy as np
import cv2

def conversion_segundos(tiempo_string):
    '''
    Convierte un string con el formato 00:01:07,800 a 67.8 (segundos)
    '''
    hhmmss = tiempo_string.split(":")
    horas = int(hhmmss[0])*60*60 # Conversión de la horas a segundos
    minutos = int(hhmmss[1])*60 # Conversion de los minutos a segundos
    segundos_milisegundos = hhmmss[2].split(",")
    segundos = float(segundos_milisegundos[0] + "." + segundos_milisegundos[1])
    tiempo = horas + minutos + segundos
    return tiempo

def obtener_tiempos(subtitulos_ruta, fps):
    '''
    Crea una lista de listas con el formato: [<tiempo_inicio>, <cantidad_frames>, <etiqueta>]
    a partir de un archivo de subtítulos y los fotogramas por segundo del video de origen.
    '''
    tiempos_frames_etiquetas = list()
    with open(subtitulos_ruta, encoding="utf8") as subtitulos:
        tiempo_0 = "00:00:00,000"
        for linea in subtitulos:
            if "-->" in linea:
                contenido = linea.split() # Separación por espacios " "
                tiempo_1 = contenido[0] # contenido[1] es -->
                tiempo_fin = contenido[2]

                tiempo_0_segundos = conversion_segundos(tiempo_0)
                tiempo_1_segundos = conversion_segundos(tiempo_1)

                if tiempo_1_segundos < tiempo_0_segundos:
                    continue

                if tiempo_1 == tiempo_0:
                    tiempo_1 = tiempo_fin

                    tiempo_0_segundos = conversion_segundos(tiempo_0)
                    tiempo_1_segundos = conversion_segundos(tiempo_1)  

                    frames_1 = int((tiempo_1_segundos - tiempo_0_segundos) * fps)
                    temp = tiempo_0.split(",")
                    tiempo_0 = temp[0] + "." + temp[1]
                    tiempos_frames_etiquetas.append([tiempo_0, str(frames_1), "1"])

                    tiempo_0 = tiempo_1
                else:
                    tiempo_fin_segundos = conversion_segundos(tiempo_fin)
                    tiempo_0_segundos = conversion_segundos(tiempo_0)
                    tiempo_1_segundos = conversion_segundos(tiempo_1)

                    frames_0 = int((tiempo_1_segundos - tiempo_0_segundos) * fps)
                    frames_1 = int((tiempo_fin_segundos - tiempo_1_segundos) * fps)

                    temp = tiempo_0.split(",")
                    tiempo_0 = temp[0] + "." + temp[1]
                    temp = tiempo_1.split(",")
                    tiempo_1 = temp[0] + "." + temp[1]

                    tiempos_frames_etiquetas.append([tiempo_0, str(frames_0), "0"])
                    tiempos_frames_etiquetas.append([tiempo_1, str(frames_1), "1"])

                    tiempo_0 = tiempo_fin

    return tiempos_frames_etiquetas

def extraer_fotogramas_etiquetados(videos_directorio):
    '''
    Genera el dataset para la red neuronal a partir de archivos mkv con subtítulos flotantes.
    Los mkv deben ser colocados en un directorio llamado videos.
    Se debe crear un directorio llamado images dentro de videos.
    Se debe crear un directorio llamado dataset dentro de videos.
    '''

    # Extracción de los fotogramas etiquetados a partir de los archivos mkv con subtítulos flotantes
    archivos_en_directorio = os.listdir(videos_directorio)
    for archivo in archivos_en_directorio:
        nombre_extension = archivo.split(".")
        if len(nombre_extension) > 1: # Evitar archivos sin extensión o directorios
            if nombre_extension[1] == "mkv": # Se analizan solo lor archivos con extension mkv
                video_ruta = os.path.join(videos_directorio, archivo)
                if nombre_extension[0] + ".srt" in archivos_en_directorio:
                    print("Los archivos de subtítulos ya existen.")
                else: # Se extraen los subtítulos en los formatos srt y ass
                    subtitulo_salida = os.path.join(videos_directorio, nombre_extension[0]) + ".srt"
                    subprocess.run(["ffmpeg.exe", "-i",
                                    video_ruta,
                                    "-vn", "-an", "-dn",
                                    subtitulo_salida,
                                    "-hide_banner"])
                    subtitulo_salida = os.path.join(videos_directorio, nombre_extension[0]) + ".ass"
                    subprocess.run(["ffmpeg.exe", "-i",
                                    video_ruta,
                                    "-vn", "-an", "-dn",
                                    subtitulo_salida,
                                    "-hide_banner"])

                if nombre_extension[0] + ".mp4" in archivos_en_directorio: # Se extraen las imágenes
                    imagenes_ruta = os.path.join(videos_directorio, "images")
                    archivos_imagen = os.listdir(imagenes_ruta)
                    video_entrada = os.path.join(videos_directorio, nombre_extension[0]) + ".mp4"
                    imagen_salida = os.path.join(videos_directorio, "images", nombre_extension[0])
                    for imagen in archivos_imagen:
                        if nombre_extension[0] not in imagen:
                            cap = cv2.VideoCapture(video_entrada)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            subtitulos_ruta = os.path.join(videos_directorio, nombre_extension[0]) + ".srt"
                            tiempos_frames_etiquetas = obtener_tiempos(subtitulos_ruta, fps)
                            contador = 0
                            for tiempo in tiempos_frames_etiquetas:
                                subprocess.run(["ffmpeg.exe", "-i",
                                                video_entrada,
                                                "-ss", tiempo[0],
                                                "-vframes", tiempo[1],
                                                imagen_salida + "_" + str(contador) + "_imagen%04d_" + tiempo[2] + ".jpg",
                                                "-hide_banner"])
                                contador += 1
                else: # Se incrustan los subtítulos ass en un archivo de salida mp4
                    video_salida = os.path.join(videos_directorio, nombre_extension[0]) + ".mp4"
                    cadena = "subtitles=./videos/" + nombre_extension[0] + ".ass"
                    subprocess.run(["ffmpeg.exe", "-i",
                                    video_ruta,
                                    "-filter_complex",
                                    cadena,
                                    video_salida,
                                    "-hide_banner"])

    # Ajuste de las imágenes al formato deseado para la red
    dataset_directorio = os.path.join(videos_directorio, "dataset")
    imagenes_directorio = os.path.join(videos_directorio, "images")                                    
    imagenes_en_directorio = os.listdir(imagenes_directorio)
    for imagen in imagenes_en_directorio:
        imagen_ruta = os.path.join(imagenes_directorio, imagen)
        img = cv2.imread(imagen_ruta, cv2.IMREAD_COLOR)
        alto, ancho, canales = img.shape # filas, columnas, profundidad
        negro = np.zeros((ancho, ancho, 3), np.uint8)
        inicio = int((ancho-alto)/2)
        negro[inicio:inicio+alto, :] = img
        img = cv2.resize(negro, (250, 250))
        # cv2.imshow("Fotograma",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        imagen_salida = os.path.join(dataset_directorio, imagen)
        cv2.imwrite(imagen_salida,img)
    
    print("Proceso finalizado.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Modo de uso: {} <videos_directorio>".format(sys.argv[0])) # argv[0] es el nombre de la función
        exit()

    extraer_fotogramas_etiquetados(sys.argv[1])