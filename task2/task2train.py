import pyspark
import findspark
from pyspark.sql import SQLContext,SparkSession
import json
import sys
import os.path
import time
from itertools import chain
import math
import functools
import heapq

from collections import Counter
from itertools import combinations

findspark.find()
spark_context = pyspark.SparkContext()
"""spark_context =  SparkSession \
    .builder \
    .master('local')\
    .appName("Task 2 HW3") \
    .getOrCreate()"""
sql_context = SQLContext(spark_context)

def dump_file(filename, data, mode='w'):
    with open(filename, mode=mode) as f:
        f.write("{}".format(data))

def get_stopwords_list(file='stopwords'):
    words_list = []
    with open(file,'r') as words:
        for word in words:
            words_list.append(word.strip())
    return words_list

def filter_split_text(iterable_text,stopwords_file):
    punctuations = '1234567890'
    stopwords = get_stopwords_list(stopwords_file)
    text = ' '.join(iterable_text).rstrip()
    text = text.translate(str.maketrans('','',punctuations)).lower()
    text = text.translate(str.maketrans('\',.()\\/_-!:;#@%^&*{}[]+=~?"№', '                            ')).lower()
    text = [word for word in text.split() if word not in stopwords and len(word)>2]
    return text

def tfidf(business_id,word,tf,idf,nwords_doc,n_doc):
    try:
        tf = tf/nwords_doc[business_id]
        idf = math.log(n_doc/(idf[word]+1))
        return tf*idf
    except KeyError:
        return 0

def get_word_in_all_docs_frequency(business_words):
    #dict of format [word]: number_of_appearences_in_documents
    doc_frequency = business_words\
                    .distinct()\
                    .map(lambda x:(x[1],x[0])) \
                    .countByKey() \
                    .items()
    return dict(doc_frequency)

def get_word_in_business_frequency(business_reviews):
    #dict of format [(business,word)]:count casted to list to parallelize late
    word_freq = business_reviews.flatMapValues(lambda x:x)\
                .countByValue()\
                .items()
    return dict(word_freq)

def save_words_indexes(unq_wordsa,file='task2.model',mode='w'):
    with open(file,mode,encoding='utf-8') as out:
        out.write("{\"word_index\":[")
        for index, word in enumerate(unq_wordsa):
            if index != len(unq_wordsa)-1:
                out.write("[\"{}\",{}],".format(word,index+1))
            else:
                out.write("[\"{}\",{}]".format(word, index+1))
        out.write("]}\n")


def save_business_profiles(business_profiles,file='task2.model',mode='w'):
    with open(file,mode,encoding='utf-8') as out:
        out.write("{\"bprofile\":")
        out.write(json.dumps(business_profiles))
        out.write("}\n")

def save_users_profiles(user_profiles,file='task2.model',mode='w'):
    with open(file,mode,encoding='utf-8') as out:
        out.write("{\"uprofile\":")
        out.write(json.dumps(user_profiles))
        out.write("}\n")

def save_user_reviews_lists(user_reviews,file='task2.model',mode='w'):
    with open(file,mode,encoding='utf-8') as out:
        out.write("{\"ureviews\":")
        out.write(json.dumps(user_reviews))
        out.write("}\n")

if __name__ == '__main__':
    start_time = time.time()
    review_file = '../train_review.json'
    output_file = 'out'
    stop_words_file = '../stopwords'


    ##Read data into triplets business text id
    data = sql_context.read.json(review_file).rdd.map(lambda x:(x['business_id'],(x['text'],x['user_id']))).partitionBy(50)

    ##convert data into business:[all reviews] pairs
    business_reviews = data.map(lambda x:(x[0],x[1][0]))\
                          .groupByKey()\
                          .map(lambda x:(x[0],filter_split_text(x[1],stop_words_file)))

    data.unpersist()
    sql_context.clearCache()
    del data

    number_of_documents = business_reviews.count()
    number_of_words_in_doc = dict(business_reviews.map(lambda x: (x[0], len(x[1]))).collect())
    business_words = business_reviews.flatMapValues(lambda x: x)

    doc_frequency = get_word_in_all_docs_frequency(business_words)
    total_words = sum(number_of_words_in_doc.values())


    business_reviews = business_reviews.map(lambda x: (x[0],[word for word in x[1] if doc_frequency[word]/total_words >= 0.0001]))
    business_reviews.persist()

    #get words tf,then ocnvert it to tf*idf
    word_freq = get_word_in_business_frequency(business_reviews)

    business_profiles_with_score = spark_context.parallelize(list(word_freq.items()))\
                          .map(lambda x:(x[0][0],(x[0][1],
                                           (x[1]/number_of_words_in_doc[x[0][0]])\
                                           *math.log(number_of_documents/doc_frequency[x[0][1]],2)))) \
                          .groupByKey() \
                          .map(lambda x: (x[0], heapq.nlargest(200, x[1], key=lambda y: y[1])))

    word_freq.clear()
    number_of_words_in_doc.clear()
    doc_frequency.clear()
    business_reviews.unpersist()
    del word_freq
    del number_of_words_in_doc
    del business_reviews

    business_profiles_without_score = business_profiles_with_score.map(lambda x: (x[0], list(set([word for word, score in x[1]])))).collect()
    save_business_profiles(business_profiles_without_score, mode='w')

    business_profiles_with_score = dict(business_profiles_without_score)
    user_profiles = sql_context.read.json(review_file).rdd.map(lambda x: (x['user_id'],x['business_id']))\
                                                           .groupByKey()\
                                                           .map(lambda x: (x[0],[business_profiles_with_score[bus] for bus in x[1]]))\
                                                           .map(lambda x:(x[0],list(chain(*x[1]))))\
                                                            .map(lambda x: (x[0], heapq.nlargest(200, x[1], key=lambda y: y[1]))) \
                                                            .map(lambda x: (x[0], list(set([word for word in x[1]])))) \
                                                            .collect()

    save_users_profiles(user_profiles,mode='a')
    user_profiles.clear()
    del user_profiles

    print("Duration: {:.2f}".format(time.time()-start_time))