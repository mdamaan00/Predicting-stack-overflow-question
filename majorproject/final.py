import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from operator import itemgetter
from collections import Counter
import numpy as np
import pandas as pd


N=45000

def cleanHtml(row):
    row=row.lower()
    clean = re.compile('<.*?>')
    cleantext = re.sub(clean, '', row)
    return cleantext

def remove_punctuation(row):   
    no_punc_txt=[]
    for ch in row:
        if ch in string.punctuation:
            no_punc_txt.append(' ')
        else:
            no_punc_txt.append(ch)
      
    no_punc_txt="".join(no_punc_txt)
        
    return no_punc_txt

def tokenize(row):
    row=row.split()
    r=[]
    for x in row:
        try:
            int(x)
        except:
            if len(x)<=10:
                r.append(x)
    return r

stopwords=stopwords.words('english')
def remove_stopwords(row):
    tokens=[]
    for word in row:
        if word not in stopwords:
            tokens.append(word)
           
    return tokens[:80]
    
p=PorterStemmer()
def stemming(row):
    stem_lst=[p.stem(word) for word in row]
    return stem_lst 




def preprocess(df):
   
    df['body']=df['body'].apply(cleanHtml)
    df['title']=df['title'].apply(cleanHtml)

    df['body']=df['body'].apply(remove_punctuation)
    df['title']=df['title'].apply(remove_punctuation)


    df['body']=df['body'].apply(tokenize)
    df['title']=df['title'].apply(tokenize)
 
    df['body']=df['body'].apply(remove_stopwords)
    df['title']=df['title'].apply(remove_stopwords)

    df['body']=df['body'].apply(stemming)
    df['title']=df['title'].apply(stemming)
    
    df['body_len']=df['body'].str.len()
    df['title_len']=df['title'].str.len()


    return df


def calculate_tf_idf_test(N,df,test):
    def freq(token):
        c=0
        if token in df:
            return df[token]
        return c
    
    tf_idf_test={}
    n=len(test)
    for i in range(n):
        tokens = test['body'][i]
        counter = Counter(tokens )
        words_count = len(tokens )
        for token in np.unique(tokens):
            tf = counter[token]/words_count
            i_df= freq(token)
            idf = np.log((N+1)/(i_df+1))
            tf_idf_test[i, token] = tf*idf
    return tf_idf_test

import pickle
vocab=pickle.load(open('./models/vocab_5000.pkl','rb'))

def vectorize(df,tf_idf):
    x_train=pd.DataFrame()
    x_train['document']=np.array(df['body'])

    for doc in df.index:

        lst=np.zeros(len(vocab))
        for i in range(len(lst)):
            if vocab[i] in x_train['document'][doc]:
                lst[i]=tf_idf[doc,vocab[i]]
        x_train['document'][doc]=lst.copy()
    return x_train


df=pickle.load(open('./models/df_5000.pkl','rb'))

def pred(dic):
    data=[[dic['title'],dic['body']]] 
    test_df=pd.DataFrame(data,columns=['title','body'])
    test_df=preprocess(test_df)
    tf_idf=calculate_tf_idf_test(N,df,test_df)
    test_df=vectorize(test_df,tf_idf)
    return test_df

