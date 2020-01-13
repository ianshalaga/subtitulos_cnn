# Compatibilidad entre Python 2 y 3
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import random
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping

def cargar_ejemplos(dataset_directorio, cantidad_ejemplos, porcentaje_prueba):
    '''
    Carga los ejemplos para usarlos en la red neuronal convolucional.
    Los ejemplos son imágenes cuadradas de 250x250.
    Los datos son normalizados: 0-255 -> 0-1
    '''
    x_entrenamiento = list()
    y_entrenamiento = list()
    x_prueba = list()
    y_prueba = list()
    lista_0 = list()
    lista_1 = list()
    cantidad_clases = 2
    imagenes_en_directorio = os.listdir(dataset_directorio)
    
    print("Armando listas de fotogramas 0 y 1.")

    for imagen in imagenes_en_directorio:
        imagen_split = imagen.split(".")
        if imagen_split[0].endswith("_0"):
            lista_0.append(imagen)
        else:
            lista_1.append(imagen)

    print("Lista de fotogramas 0 y 1 armadas.")

    cantidad_ejemplos_prueba_por_clase = int((porcentaje_prueba*cantidad_ejemplos)/cantidad_clases)
    cantidad_ejemplos_entrenamiento_por_clase = int((cantidad_ejemplos-(cantidad_ejemplos_prueba_por_clase*2))/2)
    cantidad_ejemplos_entrenamiento = cantidad_ejemplos_entrenamiento_por_clase*cantidad_clases

    print("Cargando ejemplos.")
    try:
        for imagen_0 in random.sample(lista_0, k=cantidad_ejemplos_prueba_por_clase):
            imagen_ruta = os.path.join(dataset_directorio, imagen_0)
            img = cv2.imread(imagen_ruta, cv2.IMREAD_COLOR)
            x_prueba.append(img)
            y_prueba.append(0)
            lista_0.remove(imagen_0)
        for imagen_0 in random.sample(lista_0, k=cantidad_ejemplos_entrenamiento_por_clase):
            imagen_ruta = os.path.join(dataset_directorio, imagen_0)
            img = cv2.imread(imagen_ruta, cv2.IMREAD_COLOR)
            x_entrenamiento.append(img)
            y_entrenamiento.append(0)
        for imagen_1 in random.sample(lista_1, k=cantidad_ejemplos_prueba_por_clase):
            imagen_ruta = os.path.join(dataset_directorio, imagen_1)
            img = cv2.imread(imagen_ruta, cv2.IMREAD_COLOR)
            x_prueba.append(img)
            y_prueba.append(1)
            lista_1.remove(imagen_1)
        for imagen_1 in random.sample(lista_1, k=cantidad_ejemplos_entrenamiento_por_clase):
            imagen_ruta = os.path.join(dataset_directorio, imagen_1)
            img = cv2.imread(imagen_ruta, cv2.IMREAD_COLOR)
            x_entrenamiento.append(img)
            y_entrenamiento.append(1)
    except Exception as e:
        print("¡No hay ejemplos suficientes de cada clase!")
        sys.exit(1)
    print("Ejemplos cargados.")

    x_entrenamiento = np.asarray(x_entrenamiento)
    y_entrenamiento = np.asarray(y_entrenamiento)
    x_prueba = np.asarray(x_prueba)
    y_prueba = np.asarray(y_prueba)
    '''
    Normalización de los datos:
    Las redes neuronales son dependientes de la magnitud de los datos.
    Es recomendable normalizar los datos siempre.
    '''
    print("Normalizando los ejemplos.")
    x_entrenamiento = x_entrenamiento/255.0
    x_prueba = x_prueba/255.0
    print("Ejemplos normalizados.")

    # Aleatoriza el orden de los ejemplos
    print("Aleatorizando los ejemplos de entrenamiento.")
    seed = random.random()
    random.seed(seed)
    indices_aleatorios = np.arange(cantidad_ejemplos_entrenamiento)
    random.shuffle(indices_aleatorios)
    x_entrenamiento = x_entrenamiento[indices_aleatorios]
    y_entrenamiento = y_entrenamiento[indices_aleatorios]
    print("Ejemplos de entrenamiento aleatorizados.")

    print("Cantidad de ejemplos de entrenamiento: {}:".format(x_entrenamiento.shape[0]))
    print("Cantidad de ejemplos de prueba: {}:".format(x_prueba.shape[0]))
    print("Alto (filas) de los ejemplos: {}".format(x_entrenamiento.shape[1]))
    print("Ancho (columnas) de los ejemplos: {}".format(x_entrenamiento.shape[2]))
    print("Canales de los ejemplos: {}".format(x_entrenamiento.shape[3]))

    return (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba)

def kfolding(particiones, cantidad_ejemplos, porcentaje_validacion):
    '''
    Retorna un diccionario donde la clave es el número de la partición y el valor una lista de dos elementos.
    El primer elemento de esta lista son los índices para el conjunto de entrenamiento.
    El segundo elemento de esta lista son los índices para el conjunto de validación.
    '''
    folds_dict = dict()
    cantidad_ejemplos_validacion = int(porcentaje_validacion*cantidad_ejemplos)
    paso = 1/(particiones*porcentaje_validacion)
    indices_entrenamiento = np.arange(cantidad_ejemplos)
    indices_extra = np.arange(int(cantidad_ejemplos_validacion - (cantidad_ejemplos_validacion*paso)))
    indices = np.concatenate((indices_entrenamiento, indices_extra))
    for k in range(0, particiones, 1):
        inicio = int(paso*k*cantidad_ejemplos_validacion)
        fin = int(inicio + cantidad_ejemplos_validacion)
        indices_validacion = indices[inicio:fin]
        indices_entrenamiento = list()
        for i in range(0, cantidad_ejemplos, 1):
            if i not in indices_validacion:
                indices_entrenamiento.append(i)
        indices_entrenamiento = np.asarray(indices_entrenamiento)
        folds_dict[k] = [indices_entrenamiento, indices_validacion]
    return folds_dict

def subtitulos_cnn(dataset_directorio):
    '''
    Entrena una red neuronal convolucional para detectar subtítulos inscrustados en los fotogramas de un video.
    '''

    '''VARIABLES GLOBALES'''
    CANTIDAD_EJEMPLOS = 6000
    PORCENTAJE_PRUEBA = 0.2
    PARTICIONES = 10
    CANTIDAD_EPOCAS = 100
    PORCENTAJE_VALIDACION = 0.2
    NEURONAS_SALIDA = 1
    NEURONAS_OCULTAS = 256
    CANTIDAD_CONVOLUCIONES = 3
    CANTIDAD_KERNELS = 32
    DIMENSION_CONVOLUCION = 3
    DIMENSION_POOLING = 2
    PORCENTAJE_DROPEO = 0.4
    DIMENSION_BATCH = 32
    ACTIVACION_OCULTA = "relu"
    ACTIVACION_SALIDA = "sigmoid"
    FUNCION_ERROR = "binary_crossentropy"
    OPTIMIZADOR = "adam"
    '''------------------'''

    # Carga de ejemplos
    (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cargar_ejemplos(dataset_directorio, CANTIDAD_EJEMPLOS, PORCENTAJE_PRUEBA)

    cantidad_ejemplos_entrenamiento = x_entrenamiento.shape[0]
    cantidad_ejemplos_prueba = x_prueba.shape[0]
    ejemplos_altura = x_entrenamiento.shape[1] # Filas
    ejemplos_ancho = x_entrenamiento.shape[2] # Columnas
    canales = x_entrenamiento.shape[3] # RGB

    folds_dict = kfolding(PARTICIONES, cantidad_ejemplos_entrenamiento, PORCENTAJE_VALIDACION)

    kernel_convolucion = (DIMENSION_CONVOLUCION, DIMENSION_CONVOLUCION)
    kernel_pooling = (DIMENSION_POOLING, DIMENSION_POOLING)

    suma_acierto_entrenamiento = 0
    suma_acierto_validacion = 0
    suma_error_entrenamiento = 0
    suma_error_validacion = 0
    suma_acierto_prueba = 0
    suma_error_prueba = 0

    resultados_finales = list()

    for particion in range(0, PARTICIONES, 1):
        modelo = Sequential()
        formato_entrada = (ejemplos_altura, ejemplos_ancho, canales)
        for i in range(0, CANTIDAD_CONVOLUCIONES, 1):
            modelo.add(Conv2D(input_shape=formato_entrada,
                              filters=CANTIDAD_KERNELS,
                              kernel_size=kernel_convolucion,
                              padding='same',
                              activation='relu',
                              kernel_initializer='random_uniform'))
            modelo.add(MaxPooling2D(pool_size=kernel_pooling, padding='valid'))
            nueva_dimension = int((ejemplos_altura-DIMENSION_CONVOLUCION-1)/DIMENSION_POOLING)
            formato_entrada = (nueva_dimension, nueva_dimension)

        modelo.add(Dropout(PORCENTAJE_DROPEO)) 
        modelo.add(Flatten())

        # Capa de entrada
        modelo.add(Dropout(PORCENTAJE_DROPEO))
        modelo.add(Dense(NEURONAS_OCULTAS, activation=ACTIVACION_OCULTA))

        # Capa de salida
        modelo.add(Dropout(PORCENTAJE_DROPEO))
        modelo.add(Dense(NEURONAS_SALIDA, activation=ACTIVACION_SALIDA)) 

        # Proceso de aprendizaje
        modelo.compile(optimizer=OPTIMIZADOR, # Actualización de los pesos (W)
                      loss=FUNCION_ERROR, # Función de error
                      metrics=['accuracy']) # Análisis del modelo (Tasa de acierto))

        # Detalles del modelo neuronal
        modelo.summary()

        print("Partición {}/{}".format(particion+1, PARTICIONES))

        parada_temprana = [EarlyStopping(monitor = 'val_loss',
                                         patience = 0,
                                         verbose = 2,
                                         mode = 'auto')]

        # Entrenamiento del modelo
        registro = modelo.fit(x=x_entrenamiento[folds_dict[particion][0]],
                              y=y_entrenamiento[folds_dict[particion][0]],
                              batch_size=DIMENSION_BATCH,
                              epochs=CANTIDAD_EPOCAS,
                              verbose=1,
                              callbacks=parada_temprana,
                              validation_data=(x_entrenamiento[folds_dict[particion][1]], y_entrenamiento[folds_dict[particion][1]]),
                              shuffle=True)

        # Evaluación del modelo
        error_prueba, acierto_prueba = modelo.evaluate(x_prueba, y_prueba)

        # Hacer predicciones
        # predicciones = model.predict(x_prueba)

        # Resultados de la última época del modelo
        acierto_entrenamiento = registro.history['accuracy'][-1]
        acierto_validacion = registro.history['val_accuracy'][-1]
        error_entrenamiento = registro.history['loss'][-1]
        error_validacion = registro.history['val_loss'][-1]

        resultados_finales.append([acierto_entrenamiento,
                                   acierto_validacion,
                                   error_entrenamiento,
                                   error_validacion,
                                   acierto_prueba,
                                   error_prueba])

        suma_acierto_entrenamiento += acierto_entrenamiento
        suma_acierto_validacion += acierto_validacion
        suma_error_entrenamiento += error_entrenamiento
        suma_error_validacion += error_validacion
        suma_acierto_prueba += acierto_prueba
        suma_error_prueba += error_prueba
        # Fin del for de particiones

    promedio_acierto_entrenamiento = suma_acierto_entrenamiento/PARTICIONES
    promedio_acierto_validacion = suma_acierto_validacion/PARTICIONES
    promedio_error_entrenamiento = suma_error_entrenamiento/PARTICIONES
    promedio_error_validacion = suma_error_validacion/PARTICIONES
    promedio_acierto_prueba = suma_acierto_prueba/PARTICIONES
    promedio_error_prueba = suma_error_prueba/PARTICIONES

    # Detalles de ejecución
    print("Características de los datos de entrada:")
    print("\tCantidad total de ejemplos cargados: {}".format(CANTIDAD_EJEMPLOS))
    print("\tCantidad de ejemplos de entrenamiento: {}".format(cantidad_ejemplos_entrenamiento))
    # print("\tCantidad de ejemplos de entrenamiento: {}".format(cantidad_ejemplos_entrenamiento))
    print("\tCantidad de ejemplos de prueba: {}".format(cantidad_ejemplos_prueba))
    print("\tAltura de los ejemplos (filas): {}".format(ejemplos_altura))
    print("\tAncho de los ejemplos (Columnas): {}".format(ejemplos_ancho))
    print("\tCanales de los ejemplos: {}".format(canales))
    print("Parámetros:")
    print("\tDropout: {}".format(PORCENTAJE_DROPEO))
    print("\tCantidad de convoluciones y poolings: {}".format(CANTIDAD_CONVOLUCIONES))
    print("\tCantidad de kernels: {}".format(CANTIDAD_KERNELS))
    print("\tDimensión de los kernels: {}".format(kernel_convolucion))
    print("\tDimensión del pooling: {}".format(kernel_pooling))
    print("\tNeuronas en la capa oculta: {}".format(NEURONAS_OCULTAS))
    print("\tFunción de activación en la capa oculta: {}".format(ACTIVACION_OCULTA))
    print("\tNeuronas en la capa de salida: {}".format(NEURONAS_SALIDA))
    print("\tFunción de activación en la capa de salida: {}".format(ACTIVACION_SALIDA))
    print("\tFunción de error: {}".format(FUNCION_ERROR))
    print("\tOptimizador: {}".format(OPTIMIZADOR))
    print("\tCantidad de épocas: {}".format(CANTIDAD_EPOCAS))
    print("\tDimensión batch: {}".format(DIMENSION_BATCH))
    print("\tCantidad de particiones: {}".format(PARTICIONES))
    print("Resultados del entrenamiento:")
    print("\tAcierto en el entrenamiento promedio: {}".format(promedio_acierto_entrenamiento))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en el entrenamiento en la partición {}: {}".format(i+1, resultados_finales[i][0]))
    print("\tAcierto en la validación promedio: {}".format(promedio_acierto_validacion))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en la validación en la partición {}: {}".format(i+1, resultados_finales[i][1]))
    # print("\tError en el entrenamiento promedio: {}".format(promedio_error_entrenamiento))
    # for i in range(0, len(resultados_finales), 1):
    #     print("\t\tError en el entrenamiento en la partición {}: {}".format(i+1, resultados_finales[i][2]))
    # print("\tError en la validación promedio: {}".format(promedio_error_validacion))
    # for i in range(0, len(resultados_finales), 1):
    #     print("\t\tError en la validación en la partición {}: {}".format(i+1, resultados_finales[i][3]))
    print("\tAcierto en la prueba promedio: {}".format(promedio_acierto_prueba))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en la prueba en la partición {}: {}".format(i+1, resultados_finales[i][4]))
    # print("\tError en la prueba promedio: {}".format(promedio_error_prueba))
    # for i in range(0, len(resultados_finales), 1):
    #     print("\t\tError en la prueba en la partición {}: {}".format(i+1, resultados_finales[i][5]))
    
    '''
    Dropout: sirve para evitar el sobreentrenamiento

    Activations: Funciones de activación
        elu
        softmax (para múltiples calses, una neurona por clase)
        selu
        softplus
        softsign
        relu
        tanh
        sigmoid (para dos clases, una sola neurona de salida)
        hard_sigmoid
        exponential
        linear

    Optimizers (Actualización de pesos W):
        sgd
        rmsprop
        adagrad
        adadelta
        adam
        adamax
        nadam

    Loss: Función que compara la salida deseada con la calculada. Criterio de error.
    Se busca minimizar esta función mediante el método del gradiente descendente.
        mean_squared_error
        mean_absolute_error
        mean_absolute_percentage_error
        mean_squared_logarithmic_error
        squared_hinge
        hinge
        categorical_hinge
        logcosh
        huber_loss
        categorical_crossentropy
        sparse_categorical_crossentropy
        binary_crossentropy
        kullback_leibler_divergence
        poisson
        cosine_proximity
        is_categorical_crossentropy

    Metrics: Análisis de desempeño del modelo
        accuracy
        binary_accuracy
        categorical_accuracy
        sparse_categorical_accuracy
        top_k_categorical_accuracy
        sparse_top_k_categorical_accuracy
        cosine_proximity
        clone_metric
        clone_metrics
    '''

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Modo de uso: {} <dataset_directorio>".format(sys.argv[0])) # argv[0] es el nombre de la función
    #     exit()

    # folds_dict = kfolding(10, 10, 0.2)
    # print(folds_dict)
    subtitulos_cnn(sys.argv[1])