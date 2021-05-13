import pandas as pd
import numpy as np
from osgeo import osr, gdal, ogr
import tensorflow as tf

def ObtenerPixel(Row, Column):

    DatasetNIR = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\NIR.tif")
    DatasetRED = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Red.tif")
    DatasetGREEN = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Green.tif")
    DatasetBLUE = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Blue.tif")

    DatasetNDVI = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\NDVI.tif")
    DatasetPAN = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\PAN.tif")

    DatasetSAVI01 = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Savi_L_01.tif")
    DatasetSAVI02 = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Savi_L_05.tif")
    DatasetSAVI03 = gdal.Open("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Savi_L_1.tif")

    # obtener todos los valores de pixeles (Row,Column) para las bandas solicitadas:
    #   (1) NDVI
    #   (3) Red,Green,Blue,NIR
    # ---------------------------------------------------------
    NIR_Value = DatasetNIR.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    RED_Value = DatasetRED.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    GREEN_Value = DatasetGREEN.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    BLUE_Value = DatasetBLUE.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]

    NDVI_Value = DatasetNDVI.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    PAN_Value = DatasetPAN.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]

    SAVI01_Value = DatasetSAVI01.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    SAVI02_Value = DatasetSAVI02.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    SAVI03_Value = DatasetSAVI03.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]

    # cerrar todas las imagenes datasets ... enviar None
    # -------------------------------------------------------
    DatasetNIR = None
    DatasetRED = None
    DatasetGREEN = None
    DatasetBLUE = None

    DatasetNDVI = None
    DatasetPAN = None
    DatasetRGB = None

    DatasetSAVI01 = None
    DatasetSAVI02 = None
    DatasetSAVI03 = None

    # Guardar all pixel values in a list[]
    # ----------------------------------
    PixelValues = [NDVI_Value, PAN_Value, SAVI01_Value, SAVI02_Value, SAVI03_Value, NIR_Value, RED_Value, GREEN_Value, BLUE_Value]

    # ingresar todos los pixel en lista con comma-separated string
    # -------------------------------------------------------
    OutPixelValueString = ','.join(
        [str(PixelValueString) for PixelValueString in PixelValues])

    # ------------------------------------------------------------
    #print(OutPixelValueString)

    if 'nan' in str(OutPixelValueString):
        return None
    else:
        return OutPixelValueString

def Obtener_Pixel_bands( Row, Column, ciudad):
    '''
        Esta funcion se encarga de leer las imagenes y guardar los datos en una lista
    '''
    NIR0, RED0, GREEN0, BLUE0, REAL0 = Obtener_URLS(ciudad)
    base = "C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\"
    NIR = base+NIR0
    RED = base+RED0
    GREEN = base+GREEN0
    BLUE = base+BLUE0
    REAL = base+REAL0

    DatasetNIR = gdal.Open(NIR)
    DatasetRED = gdal.Open(RED)
    DatasetGREEN = gdal.Open(GREEN)
    DatasetBLUE = gdal.Open(BLUE)
    DatasetREAL = gdal.Open(REAL)

    # obtener todos los valores de pixeles (Row,Column) para las bandas solicitadas:
    #   (1) NDVI
    #   (3) Red,Green,Blue,NIR
    # ---------------------------------------------------------
    NIR_Value = DatasetNIR.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    RED_Value = DatasetRED.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    GREEN_Value = DatasetGREEN.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    BLUE_Value = DatasetBLUE.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]
    REAL_Value = DatasetREAL.GetRasterBand(1).ReadAsArray(Column, Row, 1, 1)[0, 0]

    # cerrar todas las imagenes datasets ... enviar None
    # -------------------------------------------------------
    DatasetNIR = None
    DatasetRED = None
    DatasetGREEN = None
    DatasetBLUE = None
    DatasetREAL = None


    # Guardar all pixel values in a list[]
    # ----------------------------------
    PixelValues = [NIR_Value, RED_Value, GREEN_Value, BLUE_Value, REAL_Value]

    # ingresar todos los pixel en lista con comma-separated string
    # -------------------------------------------------------
    SalidaPixels = ','.join(
        [str(PixelValueString) for PixelValueString in PixelValues])

    # ------------------------------------------------------------
    #print(OutPixelValueString)

    if 'nan' in str(SalidaPixels):
        return None
    else:
        return SalidaPixels

def ObtenerMatriz(m1, m2, n1, n2, ciudad):
    Matriz=[]
    for i in range(m1,m2):
        Matriz.append([])
        for j in range(n1, n2):
            Matriz[i-m1].append(Obtener_Pixel_bands( i , j, ciudad ))
    return Matriz

def ObtenerDiccionario(m1, m2, n1, n2, ciudad):

    NIR_Value = []
    RED_Value = []
    GREEN_Value = []
    BLUE_Value = []
    REAL_Value = []

    Diccionario= pd.DataFrame()
    Vec=ObtenerMatriz(m1, m2, n1, n2, ciudad)

    for i in range(m1, m2):
        for j in range(n1, n2):
            s=Vec[i-m1][j-n1].split(sep=',')

            NIR_Value.append(s[0])
            RED_Value.append(s[1])
            GREEN_Value.append(s[2])
            BLUE_Value.append(s[3])
            REAL_Value.append(s[4])


    Diccionario["NIR"] = NIR_Value
    Diccionario["RED"] = RED_Value
    Diccionario["GREEN"] = GREEN_Value
    Diccionario["BLUE"] = BLUE_Value
    Diccionario["REAL"] = REAL_Value

    Xreal = pd.DataFrame()
    Xreal["NDVI"] = ObtenerNDVI(Diccionario)
    Xreal["PAN"] = ObtenerPAN(Diccionario)
    Xreal["NIR"] = pd.to_numeric(Diccionario["NIR"])
    Xreal["RED"] = pd.to_numeric(Diccionario["RED"])
    Xreal["GREEN"] = pd.to_numeric(Diccionario["GREEN"])
    Xreal["BLUE"] = pd.to_numeric(Diccionario["BLUE"])
    Xreal["Savi01"], Xreal["Savi05"], Xreal["Savi1"] = ObtenerSavi(Diccionario)

    real = pd.DataFrame()
    real["REAL"] = pd.to_numeric(Diccionario["REAL"])

    num = Contarceros(Diccionario)
    return Xreal, real, num

def ObtenerNDVI(Diccionario):
    '''
        Esta funcion calcula el indice NDVI
    '''
    nir=pd.to_numeric(Diccionario["NIR"])
    red=pd.to_numeric(Diccionario["RED"])
    Ndvi = (nir-red)/(nir+red)
    return Ndvi

def Obtner_4_Bandas(Diccionario):

    red=pd.to_numeric(Diccionario["RED"])
    green=pd.to_numeric(Diccionario["GREEN"])
    blue = pd.to_numeric(Diccionario["BLUE"])
    nir = pd.to_numeric(Diccionario["NIR"])
    return red, green, blue, nir

def ObtenerPAN(Diccionario):

    red=pd.to_numeric(Diccionario["RED"])
    green=pd.to_numeric(Diccionario["GREEN"])
    blue = pd.to_numeric(Diccionario["BLUE"])
    nir = pd.to_numeric(Diccionario["NIR"])
    PAN = (red+green+blue+nir)/4
    return PAN

def ObtenerSavi(Diccionario):

    nir=pd.to_numeric(Diccionario["NIR"])
    red=pd.to_numeric(Diccionario["RED"])
    # formula [(NIR - rojo) * (1 + L)] / (NIR + rojo + L)
    Savi01 = ((nir-red)*(1.1))/(nir+red+0.1)
    Savi05 = ((nir-red)*(1.5))/(nir+red+0.5)
    Savi1 = ((nir-red)*(2))/(nir+red+1)

    return Savi01, Savi05, Savi1

def Obtener_URLS(Ciudad):
    '''
        EN ESTA FUNCION SE DEBE ESCRIBIR LAS DIRECCIONES DE LAS IMAGENES DE LAS CIUDADES
        QUE SE QUIERAN PREESTABLECER 
    '''

    "C:\\Users\\VML0319\\Documents\\Tesis_stiven\\image\\Barrancabermeja\\Barrancabermeja_b8.tif"

    NIR0 = "Barrancabermeja\\Barrancabermeja_b8.tif"
    RED0 = "Barrancabermeja\\Barrancabermeja_b4.tif"
    GREEN0 = "Barrancabermeja\\Barrancabermeja_b3.tif"
    BLUE0 = "Barrancabermeja\\Barrancabermeja_b2.tif"
    REAL0 = "Barrancabermeja\\Barrancabermeja_real.tif"

    if Ciudad == 1:
        NIR0 = "Barrancabermeja\\Barrancabermeja_b8.tif"
        RED0 = "Barrancabermeja\\Barrancabermeja_b4.tif"
        GREEN0 = "Barrancabermeja\\Barrancabermeja_b3.tif"
        BLUE0 = "Barrancabermeja\\Barrancabermeja_b2.tif"
        REAL0 = "Barrancabermeja\\Barrancabermeja_real.tif"

    elif Ciudad == 2:
        NIR0 = "Cucuta\\Cucuta_b88.tif"
        RED0 = "Cucuta\\Cucuta_b44.tif"
        GREEN0 = "Cucuta\\Cucuta_b33.tif"
        BLUE0 = "Cucuta\\Cucuta_b22.tif"
        REAL0 = "Cucuta\\Cucuta_real.tif"

    elif Ciudad == 3:
        NIR0 = "Bogota\\Bogota_b8.tif"
        RED0 = "Bogota\\Bogota_b4.tif"
        GREEN0 = "Bogota\\Bogota_b3.tif"
        BLUE0 = "Bogota\\Bogota_b2.tif"
        REAL0 = "Bogota\\Bogota_real.tif"

    elif Ciudad == 4:
        NIR0 = "Medellin\\Medellin_b8.tif"
        RED0 = "Medellin\\Medellin_b4.tif"
        GREEN0 = "Medellin\\Medellin_b3.tif"
        BLUE0 = "Medellin\\Medellin_b2.tif"
        REAL0 = "Medellin\\Medellin_real.tif"

    elif Ciudad == 5:
        NIR0 = "Montevideo\\Montevideo_b8.tif"
        RED0 = "Montevideo\\Montevideo_b4.tif"
        GREEN0 = "Montevideo\\Montevideo_b3.tif"
        BLUE0 = "Montevideo\\Montevideo_b2.tif"
        REAL0 = "Montevideo\\Montevideo_real.tif"

  #  elif Ciudad == 6:
 #       NIR0 = "Sucre\\Sucre_b88.tif"
 #       RED0 = "Sucre\\Sucre_b44.tif"
  #      GREEN0 = "Sucre\\Sucre_b33.tif"
  #      BLUE0 = "Sucre\\Sucre_b22.tif"
  #      REAL0 = "Sucre\\Sucre_reall.tif"

   # elif Ciudad == 7:
  #      NIR0 = "Caracas\\Caracas_b8.tif"
  #      RED0 = "Caracas\\Caracas_b4.tif"
   #     GREEN0 = "Caracas\\Caracas_b3.tif"
   #     BLUE0 = "Caracas\\Caracas_b2.tif"
   #     REAL0 = "Caracas\\Caracas_real.tif"

    elif Ciudad == 6:
        NIR0 = "Incendio\\Febrero_b8.tif"
        RED0 = "Incendio\\Febrero_b4.tif"
        GREEN0 = "Incendio\\Febrero_b3.tif"
        BLUE0 = "Incendio\\Febrero_b2.tif"
        REAL0 = "Incendio\\Febrero_real.tif"

    elif Ciudad == 7:
        NIR0 = "Incendio\\Noviembre_b8.tif"
        RED0 = "Incendio\\Noviembre_b4.tif"
        GREEN0 = "Incendio\\Noviembre_b3.tif"
        BLUE0 = "Incendio\\Noviembre_b2.tif"
        REAL0 = "Incendio\\Noviembre_real.tif"

    elif type(Ciudad) == type([]):
        NIR0 = Ciudad[0]
        RED0 = Ciudad[1]
        GREEN0 = Ciudad[2]
        BLUE0 = Ciudad[3]
        REAL0 = Ciudad[4]

    return  NIR0, RED0, GREEN0, BLUE0, REAL0

def Contarceros(Diccionario):
    ceros = 0
    df = pd.to_numeric(Diccionario["NIR"])
    s = df.size
    for i in range(s):
        if int(df[i]) == 0:
            ceros=ceros+1
    return ceros