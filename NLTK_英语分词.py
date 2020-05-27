# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:03:39 2020

@author: REGGIE
"""
#%%
import nltk
from nltk.tokenize import sent_tokenize

text= "Good muffins cost $3.88\nin New York.  Please buy me two of them.\nThanks."
sent_tokenize(text)

#%%
from nltk.tokenize import word_tokenize
word_tokenize(text)

#%%
[word_tokenize(t) for t in sent_tokenize(text)]


#%%
from nltk.corpus import stopwords
english_stopwords = stopwords.words("english")

#%%
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*'] # 自定义英文表单符号列表
for i in word_tokenize("Everything is OK. I can do it all by myself."):
    if i.lower() not in english_stopwords: # 过滤停用词
        if i not in english_punctuations: # 过滤标点符号
            print(i)
            
            
#%%
#porter 模块是基于 Porter Stemming 算法的词干分析模块，
#该模块下定义了 PorterStemmer 类，通过调用该类下的 stem 方法可以实现英文词汇的词干化处理
from nltk.stem.porter import PorterStemmer
st = PorterStemmer()
words=['fishing', 'crying', 'likes', 'meant', 'owed','was', 'did', 'done', 'women',"avaliable"]
for word in words:
    print(word,st.stem(word))
    