# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:14:47 2020

@author: REGGIE
"""
#%%
import jieba
seg_list = jieba.cut(u"这是一段测试文本",cut_all = False)

print('full mode'+','.join(seg_list))

#%%
#词性标注
import jieba.posseg as pseg
words = pseg.cut('我是Reggie')
for word, flag in words:
    print('%s,%s'%(word,flag))
    
#%%
#关键词提取
import jieba.analyse
content = u'会议邀请到美国密歇根大学(University of Michigan, Ann Arbor\
环境健康科学系副教授奚传武博士作题为“Multibarrier approach for safe drinking waterin the US : \
Why it failed in Flint”的学术讲座，介绍美国密歇根Flint市饮用水污染事故的发生发展和处置等方面内容。\
讲座后各相关单位同志与奚传武教授就生活饮用水在线监测系统、\
美国水污染事件的处置方式、生活饮用水老旧管网改造、\
如何有效减少消毒副产物以及美国涉水产品和二次供水单位的监管模式等问题进行了探讨和交流。\
本次交流会是我市生活饮用水卫生管理工作洽商机制运行以来的又一次新尝试，\
也为我市卫生计生综合监督部门探索生活饮用水卫生安全管理模式及突发水污染事件的应对措施开拓了眼界和思路。'
#基于TF-IDF
keywords = jieba.analyse.extract_tags(content,topK= 6,withWeight = True)

for item in keywords:
    print(item[0],item[1])

#%%
#基于TextRank
keywords = jieba.analyse.textrank(content,topK = 5,withWeight = True)
for item in keywords:    
    print(item[0],item[1])

#%%
#测试
import os 
os.chdir(r'C:/Users/REGGIE/Desktop/NLP文本处理')

#对于一些人名和地名，jieba处理不好

jieba.suggest_freq('沙瑞金', True)                                                
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)

with open('nlp_test0.txt',encoding='UTF-8') as f:
    document = f.read()
    
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)

with open('./nlp_test1.txt', 'w') as f2:
        f2.write(result)
        print(result)
f.close()
f2.close()


with open('nlp_test2.txt',encoding='UTF-8') as f:
    document = f.read()
    
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)

with open('./nlp_test3.txt', 'w') as f3:
        f3.write(result)
        print(result)
f3.close()


#%%
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath)
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list  
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

#%%
#现在把上面分词好的文本载入内存
with open('./nlp_test1.txt') as f3:                                          
    res1 = f3.read()
print(res1)

with open('./nlp_test3.txt') as f3:                                          
    res2 = f3.read()
print(res2)    
#%%
#现在可以进行向量化，TF-IDF和标准化三步处理
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [res1,res2]
vector = TfidfVectorizer(stop_words=stpwrdlst)
tfidf = vector.fit_transform(corpus)
print(tfidf)

#获取词袋模型中的所有词  
wordlist = vector.get_feature_names()
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()  

for i in range(len(weightlist)):
    print("-------第",i,"段文本的词语tf-idf权重------"  )
    for j in range(len(wordlist)):
        print(wordlist[j],weightlist[i][j])

