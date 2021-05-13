
# In[ ]:
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import osr, gdal
import tensorflow as tf
from Funciones import *
from Entrenamiento import *
from  Imagenes import *


# Import librarys

Mostrar_bienvenida()
global Diccionario
global Xtrain
global Ytrain
global Ciudad
global Xreal
global imagen
global Lista
global Porcentaje
global nombre
global num

opcion = 1

while opcion != 0:
    opcion = opcion_menu()
    if opcion == 1:
        print("Has elegido Cargar datos de entrenamiento para la Red neuronal.")
        Datos, Objetivo = PreparararDataFromCSV()
        Xtrain, Ytrain, Xtest, Ytest = PrepararDatosentrenamiento(Datos, Objetivo)
        Resultado = Obtener_Prediccion(Xtrain, Ytrain, Xtest)
        Obtener_Verificador(Resultado, Ytest)

    elif opcion == 2:
        print("Has elegido Selecionar Ciudad.")
        Ciudad = Obtener_Ciudad()
        m1, m2, n1, n2 = Obtener_Dimension(Ciudad)
        Xreal, imagen, num = ObtenerDiccionario(m1, m2, n1, n2, Ciudad)
        nombre = Obtener_nombre(Ciudad)


    elif opcion == 3:
        print("Has elegido Insertar Imagenes.")
        nir, red, green, blue, real = Insertar_Urls()
        Ciudad = [ nir, red, green, blue, real]
        m1, m2, n1, n2 = Obtener_area()
        Xreal, imagen, num = ObtenerDiccionario(m1, m2, n1, n2, Ciudad)
        nombre = str(input("Ingrese el nombre de la ciudad : " ))

    elif opcion == 4:
        print("Has elegido Obtener prediccion.")
        Prediccion = Obtener_Prediccion(Xtrain, Ytrain, Xreal)
        Resultado, Porcentaje = Obtener_Clasificador(Prediccion)
        le = len(Resultado)
        Porcentaje = Obtener_Porcentaje(Porcentaje, le, num)
        print("Porcentaje de Vegetacion en la zona:", +Porcentaje, "%")
        Lista = Resultado

    elif opcion == 5:
        print("Has elegido Imagenes Antes y Despues.")

        m1, m2, n1, n2 = Obtener_Dimension(Ciudad)
        matrix = Lista
        Crear_Imangen_Predecida(matrix, m1, m2, n1, n2, nombre, Porcentaje)
        Crear_Imangen_Real(imagen, m1, m2, n1, n2, nombre)
        print("Imagenes creada con exito")

    elif opcion == 6:
        print("Has elegido calcular la cantidad de parques en la ciudad.")
        m1, m2, n1, n2 = Obtener_Dimension(Ciudad)
        matrix = Lista
        parques = Calcular_parques(matrix, m1, m2, n1, n2)
        print("En la ciudad de "+nombre+ " hay aproximadamente: "+str(parques)+ " parques")
        #print(Xreal)

    elif opcion == 0:
        print("Has decidido salir.")

print("Gracias por usar Clasificador Artificial. ¡Hasta pronto!")