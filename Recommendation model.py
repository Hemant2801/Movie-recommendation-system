#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# # Data collection and processing

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Movie recommendation system/movies.csv')


# In[3]:


# print first 5 rows of the dataset
df.head()


# In[4]:


# shape of the dataset
df.shape


# In[5]:


# info about the dataset
df.info()


# In[6]:


# selecting the relevant feature for recomeendation
selected_features = ['genres', 'keywords', 'title', 'runtime', 'tagline', 'cast', 'director']
selected_features


# In[7]:


# replacing the null values with null strings
for i in selected_features:
    df[i] = df[i].fillna('')


# In[8]:


# combining all the selected features
combined_features = df['genres'] +' '+ df['keywords'] +' '+ df['tagline'] +' '+ df['cast'] +' '+ df['director'] 


# In[9]:


combined_features


# In[10]:


# convert the text data to feature vectors
vectorizer = TfidfVectorizer()


# In[11]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[12]:


print(feature_vectors)


# Cosine Singularity

# In[13]:


# getting the similarity score/ similarity confidence value
similarity = cosine_similarity(feature_vectors)
similarity


# In[14]:


similarity.shape


# In[19]:


# getting the movie name from the user
movie_name = input('ENTER THE MOVIE NAME : ')


# In[17]:


# creating a list with all the movie names
movie_title_list = df['title'].tolist()


# In[18]:


movie_title_list


# In[20]:


# finding the close match to the movie name given by user
close_match_list = difflib.get_close_matches(movie_name, movie_title_list)
print(close_match_list)


# In[21]:


close_match = close_match_list[0]


# In[23]:


# finding the index of the movie
movie_index = df[df['title'] == close_match]['index'].values[0]
movie_index


# In[24]:


# getting a list of similar values
similarity_score = list(enumerate(similarity[movie_index]))
similarity_score


# In[25]:


len(similarity_score)


# In[26]:


#sorting the movies based on their similarity score
sorted_movies = sorted(similarity_score, key = lambda x : x[1], reverse = True)
sorted_movies


# In[27]:


# print the name of similar movies based on the index
print('RECOMMENDED MOVIES : ')

i = 1
for movies in sorted_movies:
    index = movies[0]
    title_name = df[df.index == index]['title'].values[0]
    if i < 21:
        print(i, ')', title_name)
        i += 1


# # Movie recommendation system

# In[23]:


# getting the movie name from the user
movie_name = input('ENTER THE MOVIE NAME : ')

# creating a list with all the movie names
movie_title_list = df['title'].tolist()

# finding the close match to the movie name given by user
close_match_list = difflib.get_close_matches(movie_name, movie_title_list)
close_match = close_match_list[0]

# finding the index of the movie
movie_index = df[df['title'] == close_match]['index'].values[0]

# getting a list of similar values
similarity_score = list(enumerate(similarity[movie_index]))

#sorting the movies based on their similarity score
sorted_movies = sorted(similarity_score, key = lambda x : x[1], reverse = True)

# print the name of similar movies based on the index
print('RECOMMENDED MOVIES : ')
i = 1
for movies in sorted_movies:
    index = movies[0]
    title_name = df[df.index == index]['title'].values[0]
    if i < 21:
        print(i, ')', title_name)
        i += 1


# In[ ]:




