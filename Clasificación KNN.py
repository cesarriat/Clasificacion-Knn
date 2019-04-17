#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importamos laslibrerias necesarias, la base datos y los modelos necesarios
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[6]:


#cargo en esta variable los datos
iris=load_iris()


# 

# In[7]:


#Averiguo q tipo de dato es, bunch es una especie de diccionario
type (iris)


# In[8]:


#descubro que tipo de datos vienen
iris.keys()


# In[9]:


#Aqui nos muestra la matriz y corresponde a los datos de la linea de arriba
#cada renglon es un flor cada columna es una medición
iris['data']


# In[10]:


#target es igual al Y de algebra o calculo, es decir es el elemento q nosotros deseamos calcular
iris['target_names']


# In[11]:


#Como es en python se arranca de cero entonces 0=seteso, 1=versicolor y 2=virginica
#fijense que la data se convierte en numero y esa es una buena practica
iris['target']


# In[12]:


#nos muestras mas detalles de las lista o atributos
iris['feature_names']


# In[13]:


#separamos la data en test y trainig, es importante usar datos para entrenar y guardase unos para validar
X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'])


# In[15]:


#tenemos 112 flores con 4 mediciones cada uno y te queda asi la matriz
X_train.shape


# In[16]:


y_train.shape


# In[18]:


#le metemos el metodo de vecinos cercanos
from sklearn.neighbors import KNeighborsClassifier


# In[22]:


#el programador define 7 que es el valor de cuantos puntos estan alrededor suyo
knn=KNeighborsClassifier(n_neighbors=7)


# In[23]:


#fit función que sirve para enterenar
knn.fit(X_train, y_train)


# In[24]:


#score funcion sirve para saber que tambien aprendio el modelo
knn.score(X_test,y_test)


# In[26]:


#ahora yo meto valores y me tiene que predecir un valor
#el valor 1 significa queesa medición pertenece a una flor versicolor
knn.predict([[1.2,3.4,5.6,1.1]])


# In[ ]:




