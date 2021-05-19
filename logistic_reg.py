# Regression logistique = c'est un algo de classification, le nom est trompeur
# Dans les algo supervisé il a deux types d'algo ceux qui vont faire de la régresssion et d'autre de la classification
# sur les regression y peut avoir une infinité de valeur en fonction de x , la valeur de y dépend de x
# Dans la regression logistique c'est pas tout a fait ça on a pas une infinité de valeur possible pour y 
# Il n'y a que deux, il resssort deux classe (vrai/faux, oui/non, 0/1)
# Sur un graphique  pour la représenter sous forme de fonction on aurait une fonction sigmuide (1/1+e^-z) donc soit 0 ou 1 avec une zone d'incertitude
# tout ce qui est en dessous de 0.5 c'est non au dessus c'est de 0.5 c'est oui
#l'exemple des nuage de point qui va déterminer la ligne de séparation le oui du non (c'est pas du 100%)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from matplotlib.colors import ListedColormap # pour graphique 
from mpl_toolkits.mplot3d import Axes3D #pour graphique
import seaborn as sns # pour 
import tensorflow as tf 

# Importer le dataset
dataset = pd.read_csv("clients.csv")

# Visualisation des données 
dataset.head()
