# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:04:45 2020

@author: REGGIE
"""
#%%
import nltk

sentence = 'I am from China !'

token = nltk.word_tokenize(sentence)

token
#%%
#中文分词
import jieba
seg_list=jieba.cut("我正在学习自然语言处理",cut_all=True)
print("全模式：","/".join(seg_list))

seg_list=jieba.cut("我正在学习自然语言处理",cut_all=False)
print("精确模式：","/".join(seg_list))

seg_list=jieba.cut("我正在学习自然语言处理")
print("默认是精确模式：","/".join(seg_list))

seg_list=jieba.cut_for_search("文本分析和自然语言处理是现在人工智能系统不可分割的一部分")
print("/".join(seg_list))

#%%
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

words = ['table', 'probably', 'wolves', 'playing', 'is', 
        'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']

#比较不同的词干提取方法
stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL']
stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmer_snowball = SnowballStemmer('english')

formatted_row = '{:>16}' * (len(stemmers) + 1)
print ('\n', formatted_row.format('WORD', *stemmers), '\n')
for word in words:
    stemmed_words = [stemmer_porter.stem(word), 
            stemmer_lancaster.stem(word), stemmer_snowball.stem(word)]
    print (formatted_row.format(word, *stemmed_words))
#结论：比较：3种词干提取算法的本质目标都是提取出词干，消除词影响。
#它们的不同之处在于操作的严格程度不同。Lancaster词干提取器比其他两个词干提取器更严格，Porter词干提取器是最宽松的。
#Lancaster词干提取器得到的词干往往比较模糊，难以理解。
#Lancaster词干提取器的速度很快，但是它会减少单词的很大部分，因此通常会选择Snowball词干提取器。


#%%
from nltk.stem import WordNetLemmatizer

words = ['table', 'probably', 'wolves', 'playing', 'is', 
        'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']

# 对比不同词形的还原器
lemmatizers = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
lemmatizer_wordnet = WordNetLemmatizer()

formatted_row = '{:>24}' * (len(lemmatizers) + 1)
print('\n', formatted_row.format('WORD', *lemmatizers), '\n')
for word in words:
    lemmatized_words = [lemmatizer_wordnet.lemmatize(word, pos='n'),
           lemmatizer_wordnet.lemmatize(word, pos='v')]
    print (formatted_row.format(word, *lemmatized_words))
    
#%%
'''
在信息检索中，为节省存储空间和提高搜索效率，在自然语言数据（或文本）之前或之后会自动过滤某些字或词，
这些字词即被称为“StopWords”。这类词基本上在任何场合任何时候都会有，
因此不会影响数据的分析结果，反而对数据处理来说，是一种多余。
因此，我们在进行分词和处理高频词汇的时候，一定要将其剔除。'''

import nltk
nltk.download('stopwords')


#%%
from sklearn.datasets import fetch_20newsgroups

#选择一个类型列表，并用词典映射的方式定义
#这些类型是加载的新闻组数据集的一部分
category_map = {'misc.forsale': 'Sales', 'rec.motorcycles': 'Motorcycles', 
        'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography', 
        'sci.space': 'Space'}
#基于刚刚定义的类型加载训练数据
training_data = fetch_20newsgroups(subset='train', 
        categories=category_map.keys(), shuffle=True, random_state=7)

#特征提取
from sklearn.feature_extraction.text import CountVectorizer

#用训练数据提取特征
vectorizer = CountVectorizer()
X_train_termcounts = vectorizer.fit_transform(training_data.data)
print("\nDimensions of training data:", X_train_termcounts.shape)

# 训练分类器
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
#定义一些随机输入的句子
input_data = [
    "The curveballs of right handed pitchers tend to curve to the left", 
    "Caesar cipher is an ancient form of encryption",
    "This two-wheeler is really good on slippery roads"
]

# tf-idf 变换器
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)

# 多项式朴素贝叶斯分类器
classifier = MultinomialNB().fit(X_train_tfidf, training_data.target)
#用词频统计转换输入数据
X_input_termcounts = vectorizer.transform(input_data)
#用tf-idf变换器变换输入数据
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

#预测输出类型
predicted_categories = classifier.predict(X_input_tfidf)

#打印输出
for sentence, category in zip(input_data, predicted_categories):
    print ('\nInput:', sentence, '\nPredicted category:', \
            category_map[training_data.target_names[category]])
    
    
#%%
#情感分析
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

 #定义一个用于提取特征的函数
def extract_features(word_list):
    return dict([(word, True) for word in word_list])


 #我们需要训练数据，这里将用NLTK提供的电影评论数据
if __name__=='__main__':
    # 加载积极与消极评论 
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')

    #将这些评论数据分成积极评论和消极评论 
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 
            'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 
            'Negative') for f in negative_fileids]

    #分成训练数据集（80%）和测试数据集（20%）
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))
     #提取特征
    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
    print ("\nNumber of training datapoints:", len(features_train))
    print ("Number of test datapoints:", len(features_test))

    #训练朴素贝叶斯分类器
    classifier = NaiveBayesClassifier.train(features_train)
    print ("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

    print ("\nTop 10 most informative words:")
    for item in classifier.most_informative_features()[:10]:
        print (item[0])

    # 输入一些简单的评论
    input_reviews = [
        "It is an amazing movie", 
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie", 
        "The direction was terrible and the story was all over the place" 
    ]
#运行分类器，获得预测结果
    print ("\nPredictions:")
    for review in input_reviews:
        print ("\nReview:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        #打印输出
        print ("Predicted sentiment:", pred_sentiment) 
        print ("Probability:", round(probdist.prob(pred_sentiment), 2))