import pandas as pd
import numpy as np
from osgeo import osr, gdal, ogr
import tensorflow as tf
import matplotlib.pyplot as plt
from osgeo import osr,gdal,gdalconst
from matplotlib.pylab import *


def Mostrar_bienvenida():
    print(" ****         ****         ****         ****         ****         ****         ****")
    print(" Bievenido a Clasificador Artificial")
    print(''' 
    Este es un pseudocódigo para dectectar vegetacion en imagenes satelitales mediante machine learning 

    Para la correcta funcionalidad de este programa se debe tener en cuenta los siguientes factores:
    1.	Tener instalada las librerías correspondientes. 
    2.	Ingresar las bandas Red, Green, Blue, y NIR de Sentinel 2 en formato Tif.
    3.	Elegir un área de procesamiento adecuada.
    ''')
    print(" ****         ****         ****         ****         ****         ****         ****")

def Insertar_imagen():
    Ubicacion= str(input("ingrese el URL con doble \\ : " ))
    return Ubicacion

def Insertar_Urls():
    print("Inserte la URL de la imagen NIR ")
    nir = Insertar_imagen()
    print("Inserte la URL de la imagen RED ")
    red = Insertar_imagen()
    print("Inserte la URL de la imagen GREEN ")
    green = Insertar_imagen()
    print("Inserte la URL de la imagen BLUE ")
    blue = Insertar_imagen()
    print("Inserte la URL de la imagen REAL ")
    real = Insertar_imagen()
    print(" Has ingresado los Urls")

    return nir, red, green, blue, real

def Elegir_Ciduad():
    ciudad= int(input("Ingrese el numero de la ciudad: " ))
    s=Obtener_nombre(ciudad)
    print("Has elegido la ciudad : "+s)
    return ciudad

def Obtener_Ciudad():

    print(''' 
        Este pseudocódigo cuenta con algunas imagenes pre-procesadas de diversas ciudades de colombia y capitales de
        latinoamerica. Puedes elegir una de las ciudades mostradas a continuacion:
        Ciudades de Colombia
        1.	Barrancabermeja
        2.	Cucuta.
        3.  Bogota no disponible, faltan imagenes
        4.	Medellin.
        Capitales de Latinoametica
        5.	Montevideo.
        Analisis de Incendios
        6.  Benalla Febrero
        7.  Benalla Noviembre
        
        ''')
    Ciudad = Elegir_Ciduad()
    #print("Inserte ubicacion de imagen NIR.")
    #IMAGE_NIR = Insertar_imagen()

    return Ciudad


def Obtener_Dimension(Ciudad):
    '''
        Esta funcion devuelve las dimensiones de las imagenes pre-establecidas,
        recibiendo el numero de la ciudad
    '''

    m1 = 1
    m2 = 10
    n1 = 1
    n2 = 10
    if Ciudad==1:
        m2 = 600
        n2 = 770
    elif Ciudad==2:
        m2 = 1020
        n2 = 1040
    elif Ciudad==3:
        m1 = 10
        m2 = 3200
        n1 = 10
        n2 = 2650
    elif Ciudad==4:
        m1 = 10
        m2 = 1570
        n1 = 10
        n2 = 1200
    elif Ciudad==5:
        m1 = 50
        m2 = 1600
        n1 = 50
        n2 = 2300
    elif Ciudad==6:
        m2 = 750
        n2 = 910
    elif Ciudad==7:
        m2 = 750
        n2 = 910
   # elif Ciudad==6:
    #    m1 = 1
    #    m2 = 820
    #    n1 = 1
    #    n2 = 920
    #elif Ciudad==7:
    #    m1 = 10
    #    m2 = 1220
    #    n1 = 10
    #    n2 = 2150

    return m1, m2, n1, n2

def Obtener_nombre(Ciudad):
    '''
        Esta funcion devuelve los nombres de la ciudades pre-establecidas
    '''

    Nombre = "Ninguna"

    if Ciudad == 1:
        Nombre = "Barrancabermeja"

    elif Ciudad == 2:
        Nombre = "Cucuta"

    elif Ciudad == 3:
        Nombre = "Bogota"

    elif Ciudad == 4:
        Nombre = "Medellin"

    elif Ciudad == 5:
        Nombre = "Montevideo"

   # elif Ciudad == 6:
    #    Nombre = "Sucre"

    #elif Ciudad == 7:
    #    Nombre = "Caracas"
    elif Ciudad == 6:
        Nombre = "Benalla Febrero"

    elif Ciudad == 7:
        Nombre = "Benalla Noviembre"

    return Nombre

def Obtener_area():

    '''
        Esta funcion recibe las dimenciones de las imagenes de una ciudad no pre-establecida
        estas imagenes deben estar en la carpeta Image
    '''

    print("Ingresar area para las Filas de la imagen")
    m1= int(input("Ingrese la fila de inicio : " ))
    m2= int(input("Ingrese la fila final: " ))
    print("Ingresar area para las Columnas de la imagen.")
    n1= int(input("Ingrese la columna de inicio : " ))
    n2= int(input("Ingrese la columna final: " ))
    return m1, m2, n1, n2

def opcion_menu():
    print("        ")
    print("Acciones disponibles:")
    print("        ")
    print("  1. Cargar Datos de entrenamiento")
    print("  2. Selecionar Ciudad de Preferencia ")
    print("  3. Insetar Imagenes")
    print("  4. Obtener prediccion")
    print("  5. Obtener Imagenes Antes y Despues")
    print("  6. Calcular cantidad de parques en la ciudad")
    print("  0. Salir")
    print("        ")
    opcion = int(input("Ingresa una opcion: "))
    while opcion < 0 or opcion > 6:
        print("No conozco la opcion³n que has ingresado. Intentalo otra vez.")
        opcion = int(input("Ingresa una opcion: "))
    return opcion

def Crear_Imangen_Predecida(Lista, m1, m2, n1, n2, ciudad, porcentaje):
    '''
        Esta funcion recibe una lista y la convierteen una una matrix 2D
        para ser transformada en una imagen y ser guardada teniendo en cuenta los pixeles
        0 pixel no arbol
        1 pixel arbol
    '''
    m = m2 - m1
    n = n2 - n1
    Matrix_imagen = np.array(Lista)
    Matrix_imagen = Matrix_imagen.reshape(m, n)
    ioff()
    plt.suptitle(' Mapa de la ciudad de '+str(ciudad))
    plt.title(' Porcentaje de vegetacion :'+str(porcentaje)+"%")
    plt.xlabel(' [ 10m/ud ] ')
    plt.ylabel(' [ 10m/ud ] ')
    #  plt.imshow( OutDataArray , cmap=plt.cm.gray )
    c = plt.imshow(Matrix_imagen, cmap='Greens' )
    plt.colorbar(c)
    plt.savefig("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Imagen_de_"+ciudad+"_Predecida.png", dpi=5000)
    plt.close()

def Crear_Imangen_Real(Lista, m1, m2, n1, n2, ciudad):
    '''
        Esta funcion recibe una lista y la convierteen una una matrix 2D
        para ser transformada en una imagen y ser guardada teniendo en cuenta los pixeles
        0 pixel no arbol
        1 pixel arbol
    '''
    m = m2 - m1
    n = n2 - n1
    Matrix_imagen = np.array(Lista)
    Matrix_imagen = Matrix_imagen.reshape(m, n)

    ioff()
    plt.title('Mapa de la ciudad de '+ciudad)
    plt.xlabel(' [ 10m/ud ] ')
    plt.ylabel(' [ 10m/ud ] ')
    c = plt.imshow(Matrix_imagen )
    plt.savefig("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Imagen_de_"+ciudad+"_Real.png", dpi=2000)
    plt.close()

def Calcular_parques( lista, m1, m2, n1, n2):
    '''
        Esta funcion Calcula la cantidad de parques teniendo en cuenta el patron establecido,
        Adicionalmente crea una imagen teniendo como referencia:
        0 pixel no arbol
        1 pixel arbol
        2 pixel posible parque

        patron numero 1
        x = parque
        0 = edificacion
        - = no importa

        -----000000000-----
        -------------------
        00---xxxxxxxxx---00
        00---xxxxxxxxx---00
        00---xxxxxxxxx---00
        00---xxxxxxxxx---00
        00---xxxxxxxxx---00
        -------------------
        -----000000000-----
    '''
    m = m2 - m1
    n = n2 - n1
    Matrix_imagen = np.array(lista)
    matrix = Matrix_imagen.reshape(m, n)
    nueva_matrix = matrix
    contador = 0
    ex = 0

    for i in range(m):
        for j in range(n):
            '''
                Patron 1 en forma condicional
            '''
            try:
                if matrix[i][j] == 1 and matrix[i + 1][j] == 1 and matrix[i - 1][j] == 1  \
                    and matrix[i][j - 1] == 1 and matrix[i][j + 1] == 1 \
                    and matrix[i][j - 2] == 1 and matrix[i][j + 2] == 1 \
                    and matrix[i + 2][j] == 1 and matrix[i - 2][j] == 1 \
                    and matrix[i - 1][j - 1] == 1 and matrix[i + 1][j + 1] == 1 \
                    and matrix[i - 1][j + 1] == 1 and matrix[i + 1][j - 1] == 1 \
                    and (matrix[i][j + 4] == 0 or matrix[i][j + 7] == 0) \
                    and (matrix[i][j - 4] == 0 or matrix[i][j - 7] == 0) \
                    and (matrix[i - 4][j] == 0 or matrix[i - 7][j] == 0) \
                    and (matrix[i + 4][j] == 0 or matrix[i + 7][j] == 0):
                    contador = contador + 1
                    nueva_matrix[i][j] = 2
                elif matrix[i][j] == 0:
                    nueva_matrix[i][j] = 0
                elif matrix[i][j] == 1:
                    nueva_matrix[i][j] = 1

            except:
                ex = ex + 1
    print(" ")
    if contador >= 1:
        "se divie la cantidad de posibles ubicaciones en 5, esperando que que haya 5 posibles ubicaciones por parque"
        contador=contador/5
        contador=int(contador)

        ioff()
        plt.title("Mapa de la ciudad de Cucuta con "+str(contador)+"parques")
        plt.xlabel(' [ 10m/ud ] ')
        plt.ylabel(' [ 10m/ud ] ')

        c = plt.imshow(nueva_matrix, cmap='Greens')
        plt.colorbar(c)
        plt.savefig("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Imagen_parques.png", dpi=5000)
        plt.close()

    return contador

def Obtener_Porcentaje(Porcentaje, len, num):
    '''
        Esta funcion obtiene el porcentaje de vegetacion de los datos predecidos
    '''
    real = (Porcentaje)/(len-num)*100
    return real

