# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:47:46 2020

@author: horna
"""
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import fashion_mnist

"""import dataset"""
(X_train, y_train),(X_test,y_test)= fashion_mnist.load_data()


"""Normalisation de 0 à 1"""
X_train = X_train/255.0
X_test = X_test/225.0
"""notre modèle de deep Learning ne vas pas trop aimer qu'on lui donne un tableau
onvatransformer ça en vecteur"""

"""onfait donc un rechape on passe d'une tableau de 28 colonneet 28 lignes
à un vecteur de 784 valeurs (28*28)"""

X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)
""" construction de notre ANN """
"""cette notre premier neuronne qui est sequentielle""" """voir la docu de 
keras"""

model= tf.keras.models.Sequential()

"""on crée notre première couche, couche d'entré dabord"""
model.add(tf.keras.layers.Dense(units=128,activation='relu', input_shape=(784, )))
"""unit=128 c'est 128 neuronnes dans la couche
input_shape= c'est la forme de nos données qui est de 784 valeurs """
model.add(tf.keras.layers.Dropout(0.2))
"""pour éviter le sur entrainement,
à chaque époque tu ajuste les poid maismais pour 20 % tu le fait pas """
model.add(tf.keras.layers.Dense(units=128,activation='relu'))"""on créee une deuxime couche ,
sans lui donnée la forme vu que l'on a deja fai, c'est la couche caché"""
model.add(tf.keras.layers.Dropout(0.2))""" pui on la dropout aussi"""
model.add((tf.keras.layers.Dense(units=10,activation='softmax'))""" on lui ouvre 
   une couche de sortie, le nombre de neuronne dans la sortie correspond 
   au nombre de catégorie qui est de 10
   softmax = tu me retourne simplement le neuronne qui a la proba la plus élevé
   par exemple pantalon 20% ou tshirt 50ù"""
model.summary()"""on revoi ce qu'il y a dans notre modèle"""

"""compilation et entrainnement"""
model.compile(optimizer='adam', loss'sparse_categorical_crossentropy', metrica=['sparse_categorical_accuracy'])

""" on doit lui définir un optimizer, ... sparse_categorical -> veritable multi classe ou du classification binaire"""

"""le modèle est compilé on peut l'entrainer"""
model.fit(X_train, y_train, epochs=25)"""epoque c'est le nbr de cycle sur lequel il va apprendre"""


"""il va ainsi à chaque époque il essaye d'optimiser son accuracy et son loss"""

"""on foit entrainer un modèle on peut le sauvegarder pour le réutiliser faut crer un dossier avant"""
filepath = 'fashion_mnist.h5'
tf.keras.models.save_model{
        model, filepath, overwrite=True, include_optimizer=True, save_format=None,
        signatures=None, option=None
        }
"""model cest le nom du fichier, overwrite c'est récrire sil existe deja tu réecris par dessus . on peut 
copier coller ce bout de code yaura plus besoin de refaire , juste de charger le modèle """














