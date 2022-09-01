#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import nltk
import re
import os

from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import lda


# In[18]:


# read scraped data 
df_more = pd.read_csv('Indeed_data.csv')
job_description = open('JD.txt').read().split('\n BREAKS HERE')
job_description = job_description[:-1]
print len(job_description)
print type(job_description[0])


# ## Tokenizing and Stemming

# Load stopwords and stemmer function from NLTK library. Stop words are words like "a", "the", or "in" which don't convey significant meaning. Stemming is the process of breaking a word down into its root.
# 

# In[4]:


# Use nltk's English stopwords.
stopwords = nltk.corpus.stopwords.words('english')

print "We use " + str(len(stopwords)) + " stop-words from nltk library."
print stopwords[:10]


# In[5]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenization_and_stemming(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
#     tokens=[]
#     for sent in nltk.sent_tokenize(text):
#         for word in nltk.word_tokenize(sent):
#             if word not in stopwords:
#                 tokens.append(word);   
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenization(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[6]:


tokenization_and_stemming("3+ years in analytics experience solving real-world business problems or doing analytics/data science research")


# Use our defined functions to analyze (i.e. tokenize, stem) our synoposes.

# In[19]:


docs_stemmed = []
docs_tokenized = []
for s in job_description:
    s = s.decode('utf-8')
    tokenized_and_stemmed_results = tokenization_and_stemming(s)
    docs_stemmed.extend(tokenized_and_stemmed_results)
    
    tokenized_results = tokenization(s)
    docs_tokenized.extend(tokenized_results)


# Create a mapping from stemmed words to original tokenized words for result interpretation.

# In[20]:


vocab_frame_dict = {docs_stemmed[x]:docs_tokenized[x] for x in range(len(docs_stemmed))}
print vocab_frame_dict['busi']


# # TF-IDF

# In[22]:


#define vectorizer parameters
tfidf_model = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenization_and_stemming, ngram_range=(1,1))

tfidf_matrix = tfidf_model.fit_transform(job_description) #fit the vectorizer to job description

print "In total, there are " + str(tfidf_matrix.shape[0]) + \
      " job postings and " + str(tfidf_matrix.shape[1]) + " terms."


# In[23]:


tfidf_model.get_params()


# Save the terms identified by TF-IDF.

# In[24]:


tf_selected_words = tfidf_model.get_feature_names()


# ## Calculate Similarity

# In[25]:


from sklearn.metrics.pairwise import cosine_similarity
cos_matrix = cosine_similarity(tfidf_matrix)
print cos_matrix


# # K-means Clustering

# In[26]:


from sklearn.cluster import KMeans

num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


# In[27]:


print len(clusters)


# ## Check K-means results

# In[29]:


# create DataFrame films from all of the input files.
np.array(clusters)  
df_more['cluster'] = pd.Series(np.array(clusters) , index=df_more.index)
#df_more = df_more.drop(['Unnamed: 0','Unnamed: 0.1'], 1)
df_more.head(10)


# In[30]:


# convert search to ints
cleanup_nums = {"Search":     {'Data+scientist': 0, 'Machine+learning engineer': 1, 'Data+analyst': 2},}
df_more.replace(cleanup_nums, inplace=True)
df_more.head()


# In[31]:


print "Number of jobs included in each cluster:"
df_more['cluster'].value_counts().to_frame()


# In[37]:


print "<Document clustering result by K-means>"

# km.cluster_centers_ denotes the importances of each items in centroid.
# need to sort it in descending order and get the top k items.
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

Cluster_keywords_summary = {}
for i in range(num_clusters):
    print "Cluster " + str(i) + " words:" ,
    Cluster_keywords_summary[i] = []
    for ind in order_centroids[i, :15]: # get the top 6 words of each cluster
        Cluster_keywords_summary[i].append(vocab_frame_dict[tf_selected_words[ind]])
        print vocab_frame_dict[tf_selected_words[ind]] + ", ",
    print 

    cluster_jobs = df_more.loc[df_more.cluster == i, 'Title'].values.tolist()
    print "Cluster " + str(i) + " titles (" + str(len(cluster_jobs)) + " jobs): " 
    #print ", ".join(cluster_jobs), '\n'


# In[35]:


print len(order_centroids[1,:])


# ## Plot result

# In[38]:


pca = decomposition.PCA(n_components=3)
tfidf_matrix_np=tfidf_matrix.toarray()
pca.fit(tfidf_matrix_np)
X = pca.transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:, 1]

#set up colors per clusters using a dict
cluster_colors = {0: 'g', 1: 'b', 2: 'r', 3: 'y', 4:'k',5:'m'}
#set up cluster names using a dict
cluster_names = {}
for i in range(num_clusters):
    cluster_names[i] = ", ".join(Cluster_keywords_summary[i])


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')

#create data frame with PCA cluster results
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 
groups = df.groupby(clusters)

# set up plot
fig, ax = plt.subplots(figsize=(16, 10))
#Set color for each cluster/group
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')

ax.legend(numpoints=1,loc=4)  #show legend with only 1 point, position is right bottom.

plt.show() #show the plot


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
search_names = ['Data+scientist', 'Machine+learning engineer', 'Data+analyst']
#create data frame with PCA cluster results
search_num = df_more['Search'].tolist()
df_indeed = pd.DataFrame(dict(x=xs, y=ys, label=search_num)) 
groups2 = df_indeed.groupby(search_num)

# set up plot
cluster_colors2 = {0: 'b', 1: 'r', 2: 'g', 3: 'y', 4:'k',5:'m'}

fig, ax = plt.subplots(figsize=(16, 10))
#Set color for each cluster/group
for name, group in groups2:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, 
            label=search_names[name], color=cluster_colors2[name], 
            mec='none')

ax.legend(numpoints=1,loc=4)  #show legend with only 1 point, position is right bottom.

plt.show() #show the plot


# # LDA

# In[63]:


#Now we use synopses to build a 100*551 matrix (terms)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=4, learning_method = 'online')

tfidf_matrix_lda = (tfidf_matrix * 100)
tfidf_matrix_lda = tfidf_matrix_lda.astype(int)


# In[64]:


lda.fit(tfidf_matrix_lda)


# <li> "model.topic_word_" saves the importance of tf_selected_words in LDA model, i.e. words similarity matrix
# <li> The shape of "model.topic_word_" is (n_topics,num_of_selected_words)
# <li> "model.doc_topic_" saves the document topic results, i.e. document topic matrix.
# <li> The shape of "model.doc_topic_" is (num_of_documents, n_topics)

# In[65]:


topic_word = lda.components_
print topic_word.shape


# In[66]:


n_top_words = 20
topic_keywords_list = []
for i, topic_dist in enumerate(topic_word):
    #Here we select top(n_top_words-1)
    lda_topic_words = np.array(tf_selected_words)[np.argsort(topic_dist)][:-n_top_words:-1] 
    for j in range(len(lda_topic_words)):
        lda_topic_words[j] = vocab_frame_dict[lda_topic_words[j]]
    topic_keywords_list.append(lda_topic_words.tolist())


# In[67]:


topic_keywords_list


# In[68]:


doc_topic = lda.transform(tfidf_matrix_lda)
print doc_topic.shape


# In[69]:


print doc_topic[:,1]


# In[70]:


topic_doc_dict = {}
titles = df_more['Title'].tolist()

print "<Document clustering result by LDA>"
for i in range(len(doc_topic)):
    topicID = doc_topic[i].argmax()
    if topicID not in topic_doc_dict:
        topic_doc_dict[topicID] = [titles[i]]
    else:
        topic_doc_dict[topicID].append(titles[i])
for i in topic_doc_dict:
    print "Cluster " + str(i) + " words: " + ", ".join(topic_keywords_list[i])
    print "Cluster " + str(i) + " titles (" + str(len(topic_doc_dict[i])) + " jobs): " 
    #print ', '.join(topic_doc_dict[i])
    print


# We got a similar set of three clusters/topics as those we got with KMeans, but the topic keywords are different.
# 
# 

# In[ ]:





# In[ ]:




