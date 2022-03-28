#!/usr/bin/env python
# coding: utf-8

# **Tenemos los siguientes datasets:**
# 
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[54]:


movies=pd.read_csv('movies.csv')  
ratings= pd.read_csv('ratings.csv')  
movies.head(5)


# In[55]:


ratings.head(5)  


# In[56]:


movies['genres'].unique()


# In[57]:


sns.countplot(ratings['rating'])


# In[58]:


plt.figure(figsize=(10,8))
ratings['rating'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)
plt.show()


# A continuación:
# 
# Datos de grupo basados ​​en: Cada usuario vio qué películas y cuánto calificó para esas películas. ID de usuario 1 ID de película vista 1, 3, 6, 47, 50 y así sucesivamente y otorgó una calificación individual para cada película.

# In[59]:


ratings=ratings[['userId','movieId','rating']] #Saco el Timestamp
ratings_df=ratings.groupby(['userId','movieId']).agg(np.max) #Agrupo los datos
ratings_df.head()


# In[60]:


count_ratings=ratings.groupby('rating').count() #Contar todos los Ratings
count_ratings


# A Continuación vemos el porcentaje para cada calificación.
# 
# calificaciones 5.0 obtuvo 13.1 % calificaciones 4.5 obtuvo 8.5 % calificaciones 4.0 obtuvo 26.6 % calificaciones 3.5 obtuvo 13.0 % calificaciones 3.0 obtuvo 19.9 % calificaciones 2.5 obtuvo 5.5 % calificaciones 2.0 obtuvo 7.5 % calificaciones 1.5 obtuvo 1.8 % calificaciones 1.0 obtuvo 2.8 % calificaciones obtuvo 1.0.5 La calificación más alta es 4.0 y la calificación más baja es 1.4

# In[61]:


count_ratings['perc_total']=round(count_ratings['userId']*100/count_ratings['userId'].sum(),1)
count_ratings


# In[62]:


# PLotting of each ratings
count_ratings['perc_total'].plot.pie()  #pie plot


# In[63]:


count_ratings['perc_total'].plot.bar()  # bar plot


# In[64]:


genres=movies['genres']
genres.head()  # visualization of genres column from movie dataset


# A continuación separamos todos los géneros de películas del conjunto de datos y use la estructura de datos "establecida" para no duplicar géneros en el mismo. 
# 
# Si en la columna "|" encuentra, separará los géneros de las películas y eliminará la duplicación en el mismo mediante el uso de la estructura de datos "establecida".

# In[65]:


genre_list=" "
for index,row in movies.iterrows():
    genre_list+=row.genres+"|"
genre_list_split=genre_list.split("|")
new_list=list(set(genre_list_split))
new_list.remove('')
new_list


# In[66]:


m=movies.copy()


# A continuación lo que quiero hacer es pasar a un valor numérico cada película a que genero pertenece pero en dimensiones separadas

# In[42]:


for genre in new_list:
    m[genre]=m.apply(lambda _:int(genre in _.genres),axis=1)
m.head()


# Hago un nuevo dataset por cada id de película con su media correspndiente y el count

# In[43]:


avg=pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean','count']))
avg 


# In[44]:


avg['movieId']=avg.index
avg # Agrego el campo moveId


# En el 70 % , la película tiene un recuento promedio de 7

# In[45]:


np.percentile(avg['count'],70)


# En el 50 %, la película tiene un recuento promedio de 3

# In[46]:


np.percentile(avg['count'],50)


# Quiero listar todas las películas y guadarlas en la variable idxtitle

# In[47]:


idx2title={int(row['movieId']):row['title']
          for _,row in movies.iterrows()}
idx2title


# In[48]:


title2idx={j:i for i,j in idx2title.items()}
title2idx


# En la pantalla anterior, todos los títulos de películas están dispuestos uno tras otro en orden inverso.

# De a cuerdo a la consigna me tengo que quedar con las películas que obtuvieron un rating >= 4

# In[49]:


highratings=ratings[ratings.rating>=4]
highratings


# Anteriormente obtuve los códigos de las películas con rating >=4 pero tamb necesito el nombre de las películas para poder recomendar y que el usuario entienda

# In[50]:


itemsets=[[idx2title[mov] for mov in highratings[highratings.userId==user].movieId]
         for user in highratings.userId]
itemsets


# Una vez que tengo los datasets armados puedo invocar al algoritmo Apriori para el sistema de recomendación.

# Lo que quiero hacer a continuación es lo siguiente:
#   - Por cada pelicula (Item) quiero poner si el usuario 1....x consumió determenida película...si la vio va a tener un True si no la vió va a tener un False.
# 
# Me transforma los items en este caso las películas y en las filas los usuarios
#     

# Si obtienen error con la librería Module not found mlxtend - la instalamos con pip install mlxtend

# In[51]:


from mlxtend.preprocessing import TransactionEncoder #pip install mlxtend
te=TransactionEncoder()
tr_ary=te.fit(itemsets).transform(itemsets)
DF=pd.DataFrame(tr_ary,columns=te.columns_)
DF.head()


# In[ ]:


from mlxtend.frequent_patterns import apriori, association_rules


# Con el parámetro use_colnames = True le estoy indicando que solo tome las columnas que tengas = True que significa lo que vio el usuario

# In[52]:


f=apriori(DF, min_support=0.2, use_colnames=True, max_len=2)
#Esto es una variante a lo que vimos la clase anterior
rules=association_rules(f,metric='lift',min_threshold=2)
rules.head()


# Sobre lo anterior, se ha recomendado que la película use el usuario donde 'antecedentes' es la columna de la película y 'consecuencias' es la columna de la película recomendada. 
# 
# En la primera fila, si ve la película '(2001:A Space Odyssey(1968))', se le recomendará '(Blade Runner(1982))' para ver más y cuya confianza de la vista es del 72 % aproximadamente. 
# 
# La columna lift se usa para decirnos que la probabilidad de ver ambas películas juntas es 2,144151 veces mayor que la probabilidad de ver solo una película. El soporte es la popularidad predeterminada de un artículo. 
# 
# En términos matemáticos, el soporte de un artículo no es más que la proporción de transacciones que involucran un artículo al número total de transacciones. support(movie_1)=(Todas las transacciones que involucran movie_1)/(total de transacciones) 
# 
# Conviction compara la probabilidad de que X aparezca sin Y si fueran dependientes con la frecuencia real de aparición de X sin Y. 
# 
# El umbral en el algoritmo Apriori identifica los conjuntos de elementos que son subconjuntos de al menos ya que cada transacción se ve como un conjunto de elementos.
# 
# El leverage calcula la diferencia entre la frecuencia observada de A y C que aparecen juntos y la frecuencia que se esperaría si A y C fueran independientes. Un valor de leverage de 0 indica independencia.

# In[ ]:




