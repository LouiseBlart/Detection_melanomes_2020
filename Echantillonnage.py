# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:15:40 2020

@author: Jeanne
"""
# modules classiques 
import numpy as np
import pandas as pd
import csv
import os 
import random
import sys

# pour la visualisation des données
import matplotlib.pyplot as plt

# modules de traitement d'images
import cv2  

# package machine learning
import sklearn 

#package TensorFlow (pour le CNN)
#import tensorflow as tf    

#package dicom
import pydicom

import shutil



np.random.seed(10)

def simple_sampling(df, size, malignancy_rate) : 
    '''
    Cette fonction crée un échantillon simple de taille fixée avec un taux de malignité fixé à partir d'un dataframe. 
    Les lignes sélectionnées dans l'échantillon sont tirées aléatoirement, de manière équiprobable. 
        
        Cette fonction :
            - prend en paramètres :
                - df : le dataframe à partir duquel on souhaite construire l'échantillon
                - size : la taille de l'échantillon   
                - malignancy_rate : le taux d'images représentant des grains de beauté malins au sein de l'échantillon
            - renvoie : l'échantillon ainsi construit
            
        Exemple : 
            df = df
            size = 100
            malignancy_rate = 0,2
            => renvoie un dataframe de 100 lignes sélectionnées aléatoirement dans df dont 20% représentent des 
            mélanomes (malins), 80% des grains de beauté bénins.  
    '''
    
    np.random.seed(10)
    
    # échantillonnage des images malignes 
    n_malin = int(malignancy_rate * size)
    df_malin = df[df["target"] == 1]
    rows_malin = np.random.choice(df_malin.index.values, n_malin, replace = False)
    df_sample_malin = df[df.index.isin(rows_malin)]
    
    # échantillonnage des images bénines
    n_benin = int((1 - malignancy_rate) * size)
    df_benin = df[df["target"] == 0]
    rows_benin = np.random.choice(df_benin.index.values, n_benin, replace = False)
    df_sample_benin = df[df.index.isin(rows_benin)]
    
    # concaténation des deux échantillons
    df_sample = df_sample_malin.append(df_sample_benin)
       
    return df_sample   