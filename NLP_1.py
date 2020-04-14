# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:27:19 2020

@author: REGGIE
"""
#%%
import nltk

nltk.download()

from nltk.book import *

#%%
#搜索文本
#搜索单词
text1.concordance('monstrous')

text2.concordance('affection')

text3.concordance('lived')

text5.concordance('lol')

#%%
#搜索相似词
text1.similar('monstrous')

text2.similar('monstrous')

#%%
#搜索共同上下文
text2.common_contexts(['monstrous','very'])

#%%
#词汇分布图
text4.dispersion_plot(['citizens','democracy','freedom','duties','America'])

#%%
#自动生成文章
text1.generate()

#%%
#计数词汇
len(text3)

sorted(set(text3))

len(set(text3))

#%%
#重复词语的密度
from __future__ import division
len(text3)/len(set(text3))

#%%
#关键词的密度
text3.count('smote')

def lexical_diversity(text):
    return len(text)/len(set(text))

def percentage(count,total):
    return 100 * count/total

#%%
lexical_diversity(text3)

percentage(text4.count('a'),len(text4))    
#%%
#词链表
 sent1=['call','me','Kodi','.']
sent1

lexical_diversity(sent1)
#%%
print(sent2)
print(sent3)

#%%
#连接
sent4+sent1

#追加
sent1.append('Cool')
print(sent1)

#索引
text4[173]

text4.index('awaken')

#切片

print(text5[16715:16735])

print(text6[1600:1625])


print(text2[141245:])

#%%
#简单统计
#频率分布
fdist1 = FreqDist(text1)
fdist1

fdist1['whale']

fdist1.plot(50,cumulative = True)

#只出现了一次的词语
fdist1.hapaxes()

#%%
#细粒度的选择词
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

#高频次而且词语长度大于7的词语
fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w)>7 and fdist5[w] > 7])

#%%
#词语搭配
from nltk.util import bigrams
list(bigrams(['more','is','said','than','done']))

text4.collocation_list()

text8.collocation_list()


#%%
#其他统计结果
[len(w) for w in text1]

#各个长度词语的出现频率
fdist = FreqDist([len(w) for w in text1])
fdist

fdist.keys()

fdist.items()

fdist.max()

fdist[3]