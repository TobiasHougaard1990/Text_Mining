""" Project coding """

"""Preprocessing"""

import os, re, glob, codecs, io
wd = 'C:\\Users\Chris\Google Drev\Studie\\7. Semester\Text-Mining\Exam'
os.chdir(wd)
import textminer as tm
dir_path = wd + '\data\\'

#Create function for reading files with different encodings
def read_multi_encode(path): 
    filenames = glob.glob(dir_path+'*.txt')
    text_ls = []
    meta_data = []
    for filename in filenames:
        try:
            text = codecs.open(filename, encoding='UTF-8').read()
        except:
            text = codecs.open(filename, encoding='LATIN-1').read()
        text_ls.append(text)
    return text_ls

#create lists with sermons
sermons = read_multi_encode(dir_path)

#Create a meta-list
def get_dates(text_list):
    date_reg = re.compile(r'% \d{6}')
    full_dates = []
    for text in text_list:
            date = date_reg.findall(text)
            full_dates.append(date)
    return full_dates

dates = get_dates(sermons)

def get_month(dates_list):
    dates = []
    for date in dates_list:
        try:
            dates.append(date[0][4:6])
        except:
            dates.append('FALSE')
    return dates

dates = get_month(dates)

#Tokenize list with sermons
tokenized_sermons=[]
for i in sermons:
     tokenized_sermons.append(tm.tokenize(i.lower()))
     
#Use pruning to remove unwanted words:
prune = tm.prune_multi(tokenized_sermons, 25, 600)
     
"""Word Counting with stopwords"""
  
#Create and apply stopword-list
filepath = wd+'\stopword_da.txt'
sw = tm.read_txt(filepath)

def texts_nosw(tokenized_texts, stopword):
    nosw = []
    for text in tokenized_texts: #For each sermon:
        nosw_text = [] #Create empty list
        nosw_text =[token for token in text if token not in stopword] #Fill the empty list with the words not in sw
        nosw.append(nosw_text)
    return nosw

sermons_nosw = texts_nosw(tokenized_sermons, sw)

#Remove s\xe5 from sermons_nosw:
def remove_s(tokenized_texts):
    nosw = []
    for text in tokenized_texts: #For each sermon:
        nosw_text = [] #Create empty list
        nosw_text =[token for token in text if token != u's\xe5'] #Fill the empty list with the words not in sw
        nosw.append(nosw_text)
    return nosw

sermons_nosw = remove_s(sermons_nosw)

from collections import Counter
words =[]
for sermon in sermons_nosw:
    for word in sermon:
        words.append(word)
        
word_count = Counter(words)
top10 = (word_count.most_common(10))
print top10

def get_rel_freq(word,words):
    rel_freq = float(word_count.get(word))/float(len(words))
    return rel_freq

relfreq=[]
for i in range (len(top10)):
    for word in top10[i][0:1]:
        relfreq.append(get_rel_freq(word,words))

print relfreq

"""Word Counting with Pruning"""
words2 =[]
for sermon in prune:
    for word in sermon:
        words2.append(word)
        
word_count2 = Counter(words2)
prune_top10 = (word_count2.most_common(10))
print prune_top10

def get_rel_freq2(word,words):
    rel_freq = float(word_count2.get(word))/float(len(words))
    return rel_freq

prune_relfreq=[]
for i in range (len(prune_top10)):
    for word in prune_top10[i][0:1]:
        prune_relfreq.append(get_rel_freq2(word,words2))

print prune_relfreq

"""Topic Modelling"""    

#Create LDA model
from gensim import corpora, models
dictionary = corpora.Dictionary(prune) #Gives each word a numerical value

#Get a list of the sermons indexed by the number of occurences of each value
sermon_bow = [dictionary.doc2bow(sermon) for sermon in prune]
del sermon

#Create Dataframe
import pandas as pd

frame = pd.DataFrame()
frame['date'] = dates
frame['date2'] = dates
frame['sermon_BOW'] = sermon_bow
  
"Slice sermons together by date"
frame.set_index(('date2'), drop=False, append=False,
                inplace=True, verify_integrity=False)#Sets index after months
frame = frame.sort_index()#resorts the lists according to the index
frame = frame.groupby(frame['date2']).sum() #Groups together variables with the same value at 'date2'
frame = frame.drop('FALSE') # removes every variable with the index-value 'FALSE'
date_clean = [1,2,3,4,5,6,7,8,9,10,11,12]
frame['date'] = date_clean #Cleans column 'month'

#Creates a LDA model from the content sermon_bow,
    #Interpreting the content with dictionary and returning k topics
k = 12
mdl = models.LdaModel(sermon_bow, id2word = dictionary,
                      num_topics = k, random_state = 1234)

#run to get the first ten words of each topic
for i in range(k):
    print 'Topic', i+1
    print [t[0] for t in mdl.show_topic(i,5)] #asks for the first 10 words of topic i
    print '---------'

"Get the topics for each month"

def topics(bow, model):
    topics = []
    for BOW in bow: #For every month run the underlying
        BOW_topic = model[BOW] #The topics of the month is placed in month_topic
        topics.append(BOW_topic) #adds the month_topic to the sermon_topics-list
    return topics

frame['Sermon_Topics'] = topics(frame['sermon_BOW'],mdl) #adds the sermon_topics to our DataFrame


"""Get the most frequent topic of each month"""
def get_frequent_topics(month):
    for result in frame['Sermon_Topics'][month]:
        if result[1] > 0.099:
            print result
            
for i in range (0,12):
    print 'Frequent topics of month',i+1,':'
    get_frequent_topics(i)
    print '-------'
