# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:15:17 2020

@author: REGGIE
"""
#%%
import nltk

from nltk.book import *

#%%
#在text2中有多少个词语，有多少不同的词语？
print(str(len(text2)))
print(str(len(set(text2))))

#%%
#不同文本之间词汇的多样性，哪一个文本中词汇更丰富？
from nltk.corpus import brown
brown.categories()
#文本一
humor_text = brown.words(categories = 'humor')

def lexical_diversity(text):
    return len(text)/len(set(text))

print(str(len(humor_text)))
print(str(len(set(humor_text))))
print('词汇多样性指标{}'.format(lexical_diversity(humor_text)))

#文本2
romance_text = brown.words(categories='romance')

print(str(len(romance_text)))
print(str(len(set(romance_text))))
print('词汇多样性指标{}'.format(lexical_diversity(romance_text)))

#%%
text5.collocation_list()

#%%
#找出文本中的所有4个字母的词，使用频率分布函数显示这些词
fdist = FreqDist([w for w in text1 if len(w) == 4])
fdist.most_common()[:20]


#%%
#定义一个函数，计算一个特定词语在文本中的出现的概率
def pre(word,text):
    pre = text.count(word)/float(len(text))
    print(str(pre*100)+'%')
    
pre('that',text5)    