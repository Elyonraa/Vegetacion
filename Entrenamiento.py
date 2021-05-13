import pandas as pd
import numpy as np
from osgeo import osr, gdal, ogr
import tensorflow as tf

def PreparararDataFromCSV():
    '''
        Esta funcion carga los datos de entrenamiento al programa y los convierte en dos matrices
    '''
    #agregar 1 para defir arboles
    """ Este docuemnto Arbol contiene informacion de pixeles que incluyen diversos tipos de vegetacion como
       arbustos, arboles y cultivos """
    Datos_entrenamiento_1 = pd.read_csv("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\Arbol.csv", header=0)
    Datos_entrenamiento_1 = Datos_entrenamiento_1.reindex( np.random.permutation(Datos_entrenamiento_1.index)    )
    a1, b1 = Datos_entrenamiento_1.shape
    a = np.ones((a1, 1), dtype=int)
    Datos_entrenamiento_1["Arbol"]=a

    """ Este docuemnto Arbol puroo contiene informacion de pixeles que solo incluyen arboles"""
    Datos_entrenamiento_2 = pd.read_csv("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\Arbol_puro_2.csv", header=0)
    Datos_entrenamiento_2 = Datos_entrenamiento_2.reindex( np.random.permutation(Datos_entrenamiento_2.index)    )
    a1, b1 = Datos_entrenamiento_2.shape
    a = np.ones((a1, 1), dtype=int)
    Datos_entrenamiento_2["Arbol"]=a

    """ Este docuemnto NO_Arbol contiene informacion de pixeles que incluyen diversos tipos de no vegetacion como
       agua, roca, casas, edificaciones """
    # agregar 0 para defir no arboles
    Datos_entrenamiento_3 = pd.read_csv("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\No_Arbol.csv", header=0)
    Datos_entrenamiento_3 = Datos_entrenamiento_3.reindex( np.random.permutation(Datos_entrenamiento_3.index)    )
    a2, b2 = Datos_entrenamiento_3.shape
    b = np.zeros((a2, 1), dtype=int)
    Datos_entrenamiento_3["Arbol"] = b

    """ Este docuemnto No Arbol puroo contiene informacion de pixeles que solo incluyen edificaciones"""
    Datos_entrenamiento_4 = pd.read_csv("C:\\Users\\VML0319\\Documents\\Tesis_stiven\\No_Arbol_puro_2.csv", header=0)
    Datos_entrenamiento_4 = Datos_entrenamiento_4.reindex( np.random.permutation(Datos_entrenamiento_4.index)    )
    a2, b2 = Datos_entrenamiento_4.shape
    b = np.zeros((a2, 1), dtype=int)
    Datos_entrenamiento_4["Arbol"] = b

    """ Esta parte unifica todos los datos de entrenamiento """
    frames = [Datos_entrenamiento_1, Datos_entrenamiento_2, Datos_entrenamiento_3, Datos_entrenamiento_4]
    Datos_entrenamiento_5 = pd.concat(frames, ignore_index=True)
    Datos_entrenamiento_5 = Datos_entrenamiento_5.reindex(np.random.permutation(Datos_entrenamiento_5.index))
    #print(training_dataframe_3)

    training_spectral_pixel_values_dataframe = Datos_entrenamiento_5.iloc[:, 0:9]
    #print(training_spectral_pixel_values_dataframe)

    arbol_dataframe = Datos_entrenamiento_5['Arbol']
    #print(type(training_spectral_pixel_values_dataframe))

    return(training_spectral_pixel_values_dataframe,arbol_dataframe)

def PrepararDatosentrenamiento(DatosX, DatosY):
    """ Esta Funcion divide los datos adquiridos en datos de entranmiento y datos de prueba"""
    Xtrain = DatosX.iloc[0:4000, :]
    Xtest = DatosX.iloc[4000: , :]
    Ytrain = DatosY.iloc[0:4000]
    Ytest = DatosY.iloc[4000: ]

    return Xtrain, Ytrain, Xtest, Ytest

def Obtener_compiled_model():
  """ Esta funciones contiene el modelo de la red neuronal del programa
  9 neuronas por cada columna
  10 capas ocultas que demostraron el mayor rendimiento en una escala de 1-20
  """
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(1 )
  ])
  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

def Obtener_compiled_model_2():
  """ Esta funciones contiene un segundo mdelo de red neuronal del programa
  las nerunas para cada capa cambiaron para combrobar rendimiento con respecto
  al primer modelo plantado
  10 capas ocultas
  """
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(7, activation="sigmoid"),
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(9, activation="sigmoid"),
    tf.keras.layers.Dense(9, activation="relu"),
    tf.keras.layers.Dense(9, activation="sigmoid"),
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(7, activation="sigmoid"),
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(1 )
  ])
  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

def Obtener_Prediccion( Xtrain, Ytrain , Xtest):
    '''
        Esta funcion se encarga de la prediccion de los datos tomando como base el
        primer modelo de red neuronal y datos de entranamiento suministrados
    '''
    dataset = tf.data.Dataset.from_tensor_slices((Xtrain.values, Ytrain.values))
    train_dataset = dataset.shuffle(len(Xtrain)).batch(1)
    model = Obtener_compiled_model()
    model.fit(train_dataset, epochs=100)

    Resultado = model.predict(Xtest.values)
    #print(Resultado)
    return Resultado

def Obtener_Clasificador(Resultado):
    '''
        Esta funcion se encarga de asignar un valor de pixel entre 0 y 1 teniendo en cuenta
        la probabilidad der ser o no ser vegetacion
    '''
    Vector = []
    len = Resultado.size
    veg = 0
    for i in range(len):
        if Resultado[i] > 0:
            Vector.append(1)
            veg = veg + 1
        else:
            Vector.append(0)
    return Vector, veg

def Obtener_Verificador(Resultado, Ytest ):
    '''
        Esta funcion se encarga de calcular la precision del modelo con respecto a los datos de prueba
    '''
    Vector = []
    len = Ytest.size
    veg =  0
    for i in range(len):
        if Resultado[i] > 0.5 :
            Vector.append(1)
        else:
            Vector.append(0)
            veg = veg +1
    #print(Vector)
    #print("comprobar")
    comprobar = []
    yee = Ytest.values
    sum = 0
    for i in range(len):
        if Vector[i] == yee[i]:
            comprobar.append(7)
        else:
            comprobar.append(0)
            # errores
            sum = sum + 1

    print(sum)
    print("Precision con de datos de prueba")
    Porcentaje = ((len - sum) / len )*100
    print("la Precision es :", +Porcentaje,"%")
    Porcentaje2 = ((len - veg) / len)*100
    print("Porcentaje de Vegetacion en la zona:", +Porcentaje2,"%")

#print("Metodo evaluate")
#print(model.evaluate(Xtrain, Ytrain))
#print("Metodo evaluate")
#print(model.evaluate(Xtrain, Ytrain))
#print(Xtest)
#print(Ytest)
#print (Datos.head())
#print (Datos.dtypes)