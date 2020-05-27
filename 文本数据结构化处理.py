# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:36:09 2020

@author: REGGIE
"""
#%%
import sklearn
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1,remove=('headers', 'footers', 'quotes'))
dataset.keys()
data_samples = dataset.data[:10] # 选取 10 条新闻文本作为分析实例
data_samples

#%%
#获得 data_samples 文档-词频矩阵
from sklearn.feature_extraction.text import CountVectorizer
dtm_vectorizer = CountVectorizer()
dtm = dtm_vectorizer.fit_transform(data_samples)
dtm
print(dtm) # 每一行都以“ （i,j） x ”的形式存在，其中，i 表示第 i 条新闻，j 表示词汇 id，x 表示词汇 j 在新闻 i 中出现的次数。

dtm.toarray()

#%%
#可以利用字典属性 vocabulary_ 查看词汇对应的 id，查看方式与字典一致。
dtm_vectorizer.vocabulary_.get("sure")

#%%
dtm_vectorizer.get_feature_names()

#%%
#inverse_transform 方法用于返回文档-词频矩阵中每个文档的非零的词项
dtm_vectorizer.inverse_transform(dtm)

#%%
from sklearn.feature_extraction.text import HashingVectorizer
hashing_vectorizer = HashingVectorizer(n_features=500)
hashing_dtm=hashing_vectorizer.fit_transform(data_samples)
hashing_dtm
hashing_dtm.toarray()

#%%
#将 data_samples 转换为 TF-IDF 矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
data_samples_tfidf=vectorizer.fit_transform(data_samples)
data_samples_tfidf
data_samples_tfidf.toarray()