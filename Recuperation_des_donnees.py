"""
Ce module permet de traiter les donnees DICOM : 
    - extraire les metadonnees
    - transformer les images .dcm en JPG
    - analyser la table
@author: louis
"""

# Pour cela nous aurons besoin des packages suivant : 
import pydicom as dicom # module pour utiliser les donnees au format DICOM
import pydicom.data
from pydicom.pixel_data_handlers.util import convert_color_space 
import matplotlib.pyplot as plt
import os # Permet d'interagir avec le systeme d'exploitation
import cv2 # Pour le traitement des image 
import PIL
import pandas as pd 
import csv
import numpy as np
import urllib.request # Permet de télécharger un fichier à partir d'une URL
import zipfile # Permet de dezipper un fichier



def Premiere_fonction ():
    global Path_Projet_Melanomes
    print(" Répondez par 1 pour 'Oui' et 0 pour 'Non' à ce questionnaire " )
    Q1= input('Avez vous déja lancé ce programme ? (le dossier "Projet_Melanomes" (contenant le fichier Base_complete (ISIC_2020_Training_Dicom dezippé), les dossiers Dicom_Sample_Test, Dicom_sample_Train, le fichier Diagnostic...) est-il deja créé ?) [1 :"oui", 0: "non"] ')
    if Q1 == '1' :
        Path_Projet_Melanomes = input( "Insérez le chemin du document 'Projet_Melanomes' (exemple : C:/Users/louis/OneDrive/Bureau/Projet_Melanomes) : ")
    elif Q1 == '0' :
        # Création du dossier Projet_Melanomes
        print ( 'Nous allons créer le dossier "Projet_Melanomes" et y télécharger le fichier contenant les données au format DICOM.')
        Path_File= input("Veuillez insérer le chemin vers lequel vous voulez créer le dossier 'Projet_Melanomes' ( exemple : C:/Users/louis/OneDrive/Bureau ) : ")
        Path_Projet_Melanomes = Path_File +'/Projet_Melanomes'
        os.mkdir(Path_Projet_Melanomes) #Creation du dossier Projet_Melanomes
        print ('Le dossier Projet_Melanomes a été créé ! ', '\n', 'Nous allons maintenant télécharger le fichier de diagnostics ','\n')
        urllib.request.urlretrieve("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv", Path_Projet_Melanomes+'/Diagnostic.csv')
        print ('Le fichier Diagnostic a été créé ! ' ,'\n', 'Nous allons également créer au préalable : ','\n','- un dossier "Dicom_Sample_Train" qui contiendra les images (au format DICOM) de notre futur échantillon training ', '\n','- un dossier "Dicom_Sample_Test" qui contiendra les images (au format DICOM) de notre futur échantillon de test '
               ,'\n','- un dossier "JPG_Sample_Train" qui contiendra les images (au format jpg) de notre futur échantillon training ', '\n','- un dossier "JPG_Sample_Test" qui contiendra les images (au format jpg) de notre futur échantillon de test ',
               '\n','- un dossier "JPG_Sample_Train_Resize" qui contiendra les images (au format jpg) de notre futur échantillon training, redimensionnées ', '\n','- un dossier "JPG_Sample_Test_Resize" qui contiendra les images (au format jpg) de notre futur échantillon de test redimensionnées ')
        Path_Dicom_Sample_Train = Path_Projet_Melanomes+'/Dicom_Sample_Train'
        Path_Dicom_Sample_Test = Path_Projet_Melanomes+'/Dicom_Sample_Test'
        Path_JPG_Sample_Train = Path_Projet_Melanomes+'/JPG_Sample_Train'
        Path_JPG_Sample_Test = Path_Projet_Melanomes+'/JPG_Sample_Test'
        Path_JPG_Sample_Train_Resize = Path_Projet_Melanomes+'/JPG_Sample_Train_Resize'
        Path_JPG_Sample_Test_Resize = Path_Projet_Melanomes+'/JPG_Sample_Test_Resize'
        os.mkdir(Path_Dicom_Sample_Train)
        os.mkdir(Path_Dicom_Sample_Test)
        os.mkdir(Path_JPG_Sample_Train)
        os.mkdir(Path_JPG_Sample_Test)
        os.mkdir(Path_JPG_Sample_Train_Resize)
        os.mkdir(Path_JPG_Sample_Test_Resize)
        # Telechargement du dossier ISIC_2020_Training_Dicom.zip 
        Q2= input("Voulez vous télécharger l'intégralité de la base sur laquelle repose ce projet (4h de téléchagement) ? (1 pour 'Oui') Sinon, ( 0 pour 'Non') il vous sera proposé de télécharger un échantillon de cette base ")
        if Q2 == '1' :
            urllib.request.urlretrieve('https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Dicom.zip', Path_Projet_Melanomes+'/ISIC_2020_Training_Dicom.zip')
            print(' Le fichier ISIC_2020_Training_Dicom.zip a bien été téléchargé, nous allons maintenant dezipper ce dossier. Cette commande prendra environ 30 minutes')
            # Dezipper de ISIC_2020_Training_Dicom.zip
            zip_ref = zipfile.ZipFile(Path_Projet_Melanomes+'/ISIC_2020_Training_Dicom.zip', 'r')
            os.mkdir(Path_Projet_Melanomes+'/Base_complete')
            zip_ref.extractall(Path_Projet_Melanomes+'/Base_complete')
            zip_ref.close()
            print("L'extraction est terminée !")
        elif Q2 == '0' : 
            print("Les commandes de ce projet ne pourront pas toutes aboutir.","\n", "\n", 
                    "Cependant, nous mettons un échantillon de cette base (et d'autres fichiers effectués à partir de la base compléte) sur le drive suivant : ","\n",
                    "https://drive.google.com/drive/folders/1ByHZayDJD6OiB7g9D3hHsUMFWFZmeBy3?usp=sharing","\n","\n", 
                    "Le dossier 'Base_complete' ne representera qu'un échantillon de la base complete, (15 fichiers dicom au lieux de plus de 33 000)","\n", 
                    "Il est dans ce cas primordial de ne pas effectuer les cellules indiquées comme 'Non conseillé en cas de téléchargement via le drive' ","\n",
                    "\n", " et de respecter les téléchargements du drive à effectuer afin de mener à bien ce projet. ","\n",
                    " Il est important de télécharger ces fichiers dans le dossier 'Projet_Melanomes' créé précedemment, et de ne pas les modifier.")
    else :
        return (" La réponse à la question doit être 1 pour 'Oui' et 0 pour 'Non'" )






class Dataframe :
    """
    Cette classe regroupe 3 fonctions : 
        - from_DICOM_to_DF : remplit un dataframe à partir des metadonnées DICOM
        - convert_DICOM_to_JPG : convertit une image DICOM en JPG
        - convert_to_JPG_RGB : convertit une image DICOM en JPG avec comme espace de couleurs le format RGB
    """
    def __init__(self):
        self.path_base_complete = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Images echantillon training/Images melanomes malins" # Chemin vers les images.dcm
        self.path_Diagnostic ="C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/Images echantillon training/Echantillon_DataFrame.csv"
        self.path_jpg = "C:/Users/louis/OneDrive/Documents/ENSAE/2A/Info/Projet melanome/test jpg" # Chemin vers les images.jpg
        self.path_jpg_RGB= "C:/Users/louis/OneDrive/Bureau/TEST/Projet_Melanomes/JPG_Sample_Train" # Chemin vers les images.jpg aux couleurs RGB
        self.columns =["image_id", "patient_age", "patient_sex", "body_part"] # informations qu'on va récupérer dans les données DICOM
        self.path_jpg_Resize = "C:/Users/louis/OneDrive/Bureau/TEST/Projet_Melanomes/JPG_Sample_Train_Resize"
        
    def from_DICOM_to_DF(self):
        '''
        Cette fonction cree et remplit un dataframe a partir de quelques-unes des informations presentes
        dans les metadonnees DICOM. 
        
        Cette fonction :
            - prend en parametre "self" permettant d'aller chercher les informations dans la partie init         
            - renvoie : le dataframe comprenant les informations extraites  
            
        Exemple : 
            w = Dataframe()
            w.from_DICOM_to_DF()
        '''
        global df
        df = pd.DataFrame(columns=self.columns)
        for file in os.listdir(self.path_base_complete ) : 
            filename = pydicom.data.data_manager.get_files(self.path_base_complete , file)[0]
            ds = pydicom.dcmread(filename)
            values = [ds.PatientID[1:-1], int(ds.PatientAge[:3]), ds.PatientSex, ds.BodyPartExamined] 
            df_new_row = pd.DataFrame(data = [values], columns = self.columns)
            df = pd.concat([df, df_new_row], ignore_index = True)
        df2 = pd.read_csv(self.path_Diagnostic) 
        # on va chercher l'information target (bénin ou malin) dans un document à part (elle n'est pas comprise dans les metadonnées DICOM)
        # ainsi que le patient_id
        df = pd.merge(df,df2[['image_name','target', "patient_id"]], left_on='image_id', right_on='image_name')
        print('Le fichier est pret !')
        return df

    def convert_DICOM_to_JPG (self) :
        '''
        Cette fonction permet de convertir un dossier dont les images sont sous
        le format DICOM en format JPG
    
        Cette fonction :
            - prend en parametres "self" permettant d'aller chercher les informations 
            dans la partie init
            - renvoie : Le dossier ou les images.jpg sont enregistrees  
            
        Exemple : 
            w = Dataframe()
            w.convert_DICOM_to_JPG()
        '''
        images_path = os.listdir(self.path_base_complete) # renvoie le nom des fichiers dans le dossier
        for n, image in enumerate(images_path):
            ds = pydicom.dcmread(os.path.join(self.path_base_complete, image)) # lire un fichier dicom a partir d'un chemin de dossier et un nom de fichier
            pixel_array_numpy = ds.pixel_array # donnees sur les pixels
            image = image.replace('.dcm', '.jpg')
            cv2.imwrite(os.path.join(self.path_jpg, image), pixel_array_numpy) # enregistrer l'image
        print ('Le dossier est pret ! ')
    
 
    def convert_to_JPG_RGB (self) : 
        ''''
        Cette fonction permet de convertir les images d'un dossier au format DICOM  
        et au format de couleurs "YBR_FULL_422", en images au format JPG et au 
        format de couleurs "RGB" (ce qui permet d'avoir un rendu plus "naturel" de l'image).
       
        Cette fonction :
            - prend en parametres "self" permettant d'aller chercher les informations 
            dans la partie init
            - renvoie : Le dossier ou les images.jpg et rgb sont enregistrees  
    
        Exemple : 
            w = Dataframe()
            w.convert_to_JPG_RGB()
        '''    
        images_path = os.listdir(self.path_base_complete) 
        for n, image in enumerate(images_path) : 
            ds = pydicom.dcmread(os.path.join(self.path_base_complete, image))
            convert = convert_color_space(ds.pixel_array, 'YBR_FULL_422', 'RGB')
            image = image.replace('.dcm', '.jpg') 
            cv2.imwrite(os.path.join(self.path_jpg_RGB, image), cv2.cvtColor(convert, cv2.COLOR_RGB2BGR))
        print ('Le dossier est pret !')
        
        
    def redimensionner (self, size) : 
        """
        Cette fonction permet de redimensionner les images du dossier path_jpg
        Cette fonction prend en paramètre : 
            - le parametre 'self' permettant de se référer aux caractéristiques présentes dans la partie initialisation
            - le parametre 'size' : un tuple indicant les dimensions souhaitées pour l'image
        """
        for file in os.listdir(self.path_jpg_RGB) :
            im = cv2.imread(self.path_jpg_RGB + '/'+ file)
            im=cv2.resize(im,size)
            cv2.imwrite(self.path_jpg_Resize+ '/'+ file, im)
        print ('Le dossier est pret !')

#w = Dataframe()
#w.convert_to_JPG_RGB()
