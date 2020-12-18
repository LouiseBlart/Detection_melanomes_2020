# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:28 2020

@author: louis
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from keras.models import load_model
from keras.preprocessing.image import (load_img ,img_to_array)

class_names = ['benin', 'malin']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}


def complexite_couleurs (image): 
    '''
    Cette fonction prend en paramétre une image. Elle affiche un 
    graphique représentant les différents canaux de couleurs en fonction
    que cette image soit en couleur ou en nuances de gris. 
    Elle montre ainsi la différence de complexité entre les deux formats.
    
    Elle prend en parametre : 
        - image : le chemin vers une image 
        
    Elle retourne : 
        - Un graphe représentant les canaux de couleurs 
        - Un graphe représentant le canal de nuance de gris
    '''
    
    img1 = cv2.imread(image,cv2.IMREAD_COLOR)
    color = ('b','g','r')
    plt.subplot(2,1,2)
    for i,col in enumerate(color):
        histr = cv2.calcHist([img1],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
    print ( "Trois canaux représentant les espaces de couleurs Rouge, Vert et Bleu")
    plt.show()
    img2= cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    roihist = cv2.calcHist([img2],[0], None, [256], [ 0, 256] )
    plt.subplot(2,1,1)
    xs=np.linspace(0,255,256)
    plt.plot(xs,roihist,color='k')
    print( ' Un seul canal pour les images en nuances de gris')
    
def display_examples(class_names, images, labels):
    '''
    Cette fonction permet de visualiser un échantillon d'images de la base "images" 
    accompagnées de leur label ('bénin' ou 'malin'). 
    
    Elle prend en paramétre : 
        - class_names : classe selon laquelle nous répartissons les images
        - images : base de données 
        - labels : les labels associés à la base "images"
    
    Exemple : 
        display_examples(class_names, train_images, train_labels)
    '''
   
    fig = plt.figure(figsize=(15,10))
    fig.suptitle("Exemples d'images de la base train", fontsize=20)
    for i in range(15):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i].astype(np.integer)])
    plt.show()
    