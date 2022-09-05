#!/usr/bin/env python
# coding: utf-8

# # Classify Data Science Related Jobs on Indeed

# In this project, I am going to use Web Scraping techniques to collect job posting data on Indeed.com, and then try to classify the job postings using Clustering based on the job descriptions.   

# ## Part 1: web scraping

# I am going to get job listings from Indeed.com using BeautifulSoup. Luckily, Indeed.com is a simple text page where we can easily find relevant entries.

# In[1]:


base_url = "http://www.indeed.com"    

URL = "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l=New+York&start=10"


# In[2]:


import urllib
import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
import os.path
print("hello")
CWD = os.path.abspath(os.path.dirname(sys.executable))


with open(os.path.join(CWD, "config.ini")) as config_file:
    print(config_file.read())
   
try:
    # >3.2
    from configparser import ConfigParser
except ImportError:
    # python27
    # Refer to the older SafeConfigParser as ConfigParser
    from configparser import SafeConfigParser as ConfigParser

config = ConfigParser()

# get the path to config.ini
config_path = os.path.join(CWD, "config.ini")

# check if the path is to a valid file
if not os.path.isfile(config_path):
    raise BadConfigError # not a standard python exception

config.read(config_path)


# Get a list of all sections
print('Sections: %s' % config.sections())

# You can treat it as an iterable and check for keys
# or iterate through them
if 'main_section' not in config:
    print('Main section does exist in config.')

#for section in config:
#    print('Section: %s' % section)
#    for key, value in config[section].items():
#        print('Key: %s, Value: %s' % (key, value))

# If you know exactly what key you are looking for,
# try to grab it directly, optionally providing a default
#print(config['main_section'].get('key1'))  # Gets as string
#print(config['main_section'].getint('key2',))
#print(config['main_section'].getfloat('key3'))
#print(config['main_section'].getboolean('key99', False))



# In[3]:
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
html = requests.get(URL, headers=headers)

# show a picture


# While this has some more verbose elements removed, we can see that there is some structure to the above:
# 
# 1. The title of a job is in a link with class set to jobtitle and a data-tn-element="jobTitle.
# 2. The location is set in a span with class='location'.
# 3. The company is set in a span with class='company'.
# 
# ![page%20source.png](attachment:page%20source.png)

# ### Extract location, company, job title and summary of job posting
# 

# In[5]:


# urls = soup.findAll('a',{'rel':'nofollow','target':'_blank'}) #this are the links of the job posts
# urls = [link['href'] for link in urls] 
    
# print urls[0]


# In[38]:


# function to get above information
df = pd.DataFrame(columns=["Title","Search","Location","Company", "detail_url"])

# search is the input term
def parse(url, df, search):
    html = requests.get(url, headers=headers)
    soup = BeautifulSoup(html.content, 'html.parser', from_encoding="utf-8")    
    for each in soup.find_all(class_= "result" ):
        try: 
            title = each.find(class_='jobTitle').text.replace('\n', '')
        except:
            title = 'None'
        try:
            location = each.find(class_= "companyLocation").text.replace('\n', '')
        except:
            location = 'None'
        try: 
            company = each.find(class_='companyName').text.replace('\n', '')
        except:
            company = 'None'
        try:
            detail_url = "www.indeed.com" + each.a['href']
        except: 
            detail_url = 'None'

        df_new_row = pd.DataFrame({'Title':title, "Search":search, 'Location':location, 'Company':company, 'detail_url':detail_url}, index=[0])
        df = pd.concat([df, df_new_row])
    return df


# In[7]:


parse(URL,df, "data scientist")


# #### filter ads and expand the data

# In[94]:


import numpy as np
import pandas as pd
import nltk
import re
import os

from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt



# Now, to scale up our scraping, we need to accumulate more results. We can do this by examining the URL above.
# 
# "http://www.indeed.com/jobs?q=data+scientist+%2420%2C000&l=New+York&start=10"
# There are two query parameters here we can alter to collect more results, the l=New+York and the start=10. The first controls the location of the results (so we can try a different city). The second controls where in the results to start and gives 10 results (thus, we can keep incrementing by 10 to go further in the list).
# 
# 
# 

# In[34]:


# create a template for changing search term, city and number of postings. 
# url_template.format(search, city, start)
url_template = "http://www.indeed.com/jobs?q={}&l={}&start={}"   
YOUR_CITY = 'Washington%2C+DC'
max_results_per_city = config['main_section'].getint('maxresultspercity') # Set this to a high-value (5000) to generate more results.
search_term = config['main_section'].get('searchterms').split(', ')
cities = config['main_section'].get('cities').split(', ')
#search_term = set(['Data+scientist', 'Machine+learning engineer', 'Data+analyst'])
#cities = set(['New+York', 'Chicago', 'San+Francisco', 'Austin', 'Seattle',
#    'Los+Angeles', 'Philadelphia', 'Atlanta', 'San+Jose', YOUR_CITY])
# cities = set(['New+York', 'Chicago', 'San+Francisco', 'Austin', 'Seattle', 
#     'Los+Angeles', 'Philadelphia', 'Atlanta', 'Dallas', 'Pittsburgh', 
#     'Portland', 'Phoenix', 'Denver', 'Houston', 'Miami', YOUR_CITY, 
#     'Charlottesville', 'Richmond', 'Baltimore', 'Harrisonburg', 'San+Antonio', 'San+Diego', 'San+Jose'
#     'Austin', 'Jacksonville', 'Indianapolis', 'Columbus', 'Fort+Worth', 'Charlotte', 'Detroit', 'El+Paso', 
#     'Memphis', 'Boston', 'Nashville', 'Louisville', 'Milwaukee', 'Las+Vegas', 'Albuquerque', 'Tucson', 
#     'Fresno', 'Sacramento', 'Long+Beach', 'Mesa', 'Virginia+Beach', 'Norfolk', 'Atlanta', 'Colorado+Springs',
#     'Raleigh', 'Omaha', 'Oakland', 'Tulsa', 'Minneapolis', 'Cleveland', 'Wichita', 'Arlington', 'New+Orleans', 
#     'Bakersfield', 'Tampa', 'Honolulu', 'Anaheim', 'Aurora', 'Santa+Ana', 'Riverside', 'Corpus+Christi', 'Pittsburgh', 
#     'Lexington', 'Anchorage', 'Cincinnati', 'Baton+Rouge', 'Chesapeake', 'Alexandria', 'Fairfax', 'Herndon',
#     'Reston', 'Roanoke'])


# In[39]:


df_more = pd.DataFrame(columns=["Title","Search","Location","Company", "detail_url"])

for search in search_term:
    for city in cities:
        print(search + ' '+ city)
        for start in range(0, max_results_per_city, 10):
            url = url_template.format(search, city, start)
            #print(url)
            df_more = parse(url,df_more, search)
            #print(df_more)


# In[40]:


print(df_more.shape)
df_more.head(100)


# In[42]:


df_more=df_more.dropna().drop_duplicates()
print('You have ' + str(df_more.shape[0]) + ' results. ')


# In[45]:


df_more = df_more[~df_more['Location'].isin(['None'])]
#print(df_more.shape)
df_more.describe()


# In[74]:


df_more.shape


# ### read url

# In[59]:

"""
# save useful job description into a list
def parse_jd(url):
    jd = 'None'
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser', from_encoding="utf-8")
    for each in soup.find_all(class_="jobsearch-JobComponent-description icl-u-xs-mt--md"):        
        jd =  each.text.replace('\n', '')
    return jd


# In[60]:


url_detail = df_more['detail_url']
summary = []
check_progress = 0
for url in url_detail:
    if check_progress%50 == 0:
        print(check_progress)
    check_progress += 1
    url_new = base_url + url
    try:
        s = parse_jd(url_new)
        summary.append(s) 
    except: 
        continue
        
    


# In[80]:


print(len(summary))
summary[0]


# In[64]:


textfile = open('JD.txt', 'w')
check_progress = 0
for s in summary:
    #if check_progress%50 == 0:
        #print check_progress/50*('*') 
    check_progress += 1
    s = s.encode('utf-8')
    textfile.write(s + '\n')
    textfile.write('\n BREAKS HERE')
textfile.close()


# read data from txt

# In[69]:


job_description = open('JD.txt').read().split('\n BREAKS HERE')
print(len(job_description))

"""
# In[71]:
import datetime
current_datetime = datetime.datetime.now()
str_date = current_datetime.strftime("%m-%d-%Y")
df_more.to_csv(os.path.join(CWD, "Indeed_data_"+str_date+".csv"), encoding='utf-8')

