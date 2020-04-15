# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:35:17 2020

@author: REGGIE
"""
#%%
import nltk
nltk.corpus.gutenberg.fileids()

#%%
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print(int(num_chars/num_words),int(num_words/num_sents),int(num_words/num_vocab),fileid)
    
#%%
sentences = gutenberg.sents('shakespeare-macbeth.txt')
sentences
sentences[1037]
#最长的句子
long = max([len(s) for s in sentences])
[s for s in sentences if len(s) == long]

#%%
#网络和聊天文本
from nltk.corpus import webtext
webtext.fileids()
for fileid in webtext.fileids():
    print(fileid,webtext.raw(fileid)[:60])

#%%
#布朗语料库
from nltk.corpus import brown
brown.categories()
brown.words(categories='news')

news_words = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_words])
modals = ['can','could','may','might','must','will']

for m in modals:
    print(m,fdist[m]) 
         
         