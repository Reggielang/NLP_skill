# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:54:57 2020

@author: REGGIE
"""
#%%
import sklearn
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1,remove=('headers', 'footers', 'quotes'),categories=['rec.sport.baseball'])

#%%构建 dataset.data 文档-词项矩阵
from sklearn.feature_extraction.text import CountVectorizer
dtm_vectorizer = CountVectorizer(stop_words="english")
dtm = dtm_vectorizer.fit_transform(dataset.data).toarray()
dtm

#%%在文档-词项矩阵 dtm 基础上构建 TF-IDF 矩阵
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(dtm).toarray()
tfidf

#%%在训练词向量之前先进行分词和停用词过滤等处理
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
english_stopwords = stopwords.words("english")
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*'] # 自定义英文表单符号列表
words=[word_tokenize(t) for t in dataset.data]
words_lower=[[j.lower() for j in i] for i in words] # 小写处理
words_clear=[]
for i in words_lower:
    words_filter=[]
    for j in i:
        if j not in english_stopwords:# 过滤停用词
            if j not in english_punctuations:# 过滤标点符号
                words_filter.append(j)
    words_clear.append(words_filter)
words_clear

#%%使用初步处理后的词汇列表 words_clear 训练词向量
import gensim
model = gensim.models.Word2Vec(words_clear, size=100, window=5, min_count=5)

#%%查看模型中所有词汇
set(model.wv.vocab.keys())

#%%查看训练得到的词向量
model['sorry']