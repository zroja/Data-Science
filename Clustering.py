
# coding: utf-8

# In[1]:


import csv


# In[5]:


with open("C:/Users/CSTEP LC/Downloads/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt", "r") as text_file:
    lines = text_file.read().split('\n')

lines = [line.split("\t") for line in lines if len(line.split("\t"))==2 and line.split("\t")[1]]


# In[6]:


train_documents = [line[0] for line in lines]


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[9]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
train_documents= tfidf_vectorizer.fit_transform(train_documents)


# In[10]:


from sklearn.cluster import KMeans


# In[11]:


km = KMeans(n_clusters = 3, init = 'k-means++', max_iter =100, n_init = 1, verbose = True)
km.fit(train_documents)


# In[13]:


count = 0
for i in range(len(lines)):
    if count>3:
        break
    if km.labels_[i]==2:
        print(lines[i])
        count+=1

