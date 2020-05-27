# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:27:53 2020

@author: REGGIE
"""
#%%
import os 
os.chdir(r'C:\Users\REGGIE\Documents\GitHub\NLP_skill')
#%%
import jieba
seg_list = jieba.cut('我爱文本数据分析')

for i in jieba.cut("我爱文本数据分析"):
    print(i)
#%%
list(jieba.cut("我爱文本数据分析"))

#%%
#jieba.cut_for_search 是适用于搜索引擎构建倒排索引（Inverted index）的分词函数，调用方式为：jieba.cut_for_search(sentence, HMM=True)
for i in jieba.cut_for_search("我爱文本数据分析"):
    print(i)
    
#%%
jieba.lcut("我爱文本数据分析")
#%%
jieba.lcut_for_search("我爱文本数据分析")

#%%
#可以包含 jieba 自带词库里没有的词汇，从而保证更高的正确率。
jieba.load_userdict('sample.txt')

#%%
###jieba.add_word 函数用于给词典中增加新的词汇，调用方式为：jieba.add_word(word, freq=None, tag=None)
###jieba.del_word 函数用于删除词典中的词汇，调用方式为：jieba.del_word(word)

#%%
###jieba.suggest_freq 函数用于调节词语的词频，用于将一个词汇中的字符分开或者合并以增加该切分出该词汇的可能性。

#%%
#posseg 包是 jieba 中实现词性标注功能的包
from jieba import posseg
pos = list(jieba.posseg.cut("我爱文本数据分析"))
for i in pos:
    print(i)

#%%
jieba.posseg.lcut("我爱文本数据分析")