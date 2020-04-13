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
