# -*- coding: utf-8 -*-
"""
Created on Tue  26 11:47:49 2020

@author: REGGIE
"""
#%%
from jieba.analyse import *
import os
os.chdir(r'C:\Users\REGGIE\Documents\GitHub\NLP_skill')

with open('sample.txt',encoding='utf-8') as f:
    data = f.read()

for keyword,weight in extract_tags(data, topK=10,withWeight=True):
    print('{} {}'.format(keyword, weight))

#%%
for keyword, weight in textrank(data, withWeight=True):
    print('{} {}'.format(keyword, weight))