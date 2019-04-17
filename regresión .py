#!/usr/bin/env python
# coding: utf-8

# Cargamos las librerias necesarias y la base de datos

# In[3]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge


# In[5]:


boston = load_boston()


# In[6]:


boston.keys()


# tenemos de la base de datos 506 casas con 13 caractristicas  atributos por cada casa

# In[14]:



boston.data.shape


# In[15]:


X_ent, X_test, y_ent, y_test=train_test_split(boston.data,boston.target)


# In[16]:


X_ent.shape
X_test.shape


# Vamos a usar 379 datos para entrenar( usandoo _ent) y 127( usando _test) datos para que una vez aplicado el modelo veer si estan bien el modelo propuesto

# In[19]:


knn = KNeighborsRegressor(n_neighbors=3)


# In[20]:


knn.fit(X_ent,y_ent)


# ¿Que tambien aprendio el modelo?

# In[21]:


knn.score(X_test, y_test)


# El valor 0.52 es muy malo, un buen valor seria cercano a 1, para ello se propone cambiar el numero de vecinos cercanos, se habia colocado inincialmente un valor de 3

# Aplicamos otro modelo  el de regresion y con .score vemos si es cercano a uno cuan bueno es

# In[22]:


rl = LinearRegression()


# In[23]:


rl.fit(X_ent,y_ent)


# In[24]:


rl.score(X_test,y_test)


# Aplicamos el modelo Ridge

# In[27]:


ridge = Ridge(alpha=.2)
ridge.fit(X_ent,y_ent)
ridge.score(X_test,y_test)


# In[ ]:




